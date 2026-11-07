package greenquanta;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.power.models.PowerModelHostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerTimeShared;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import ch.qos.logback.classic.Level;
import org.cloudsimplus.util.Log;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * GreenQuanta Phase 1: replay 24 hours of real Google 2019 CPU demand in CloudSim Plus,
 * with VMs placed exactly where the Quantum Annealing optimizer decided.
 *
 * Inputs (written by optimizer/run_phase1.py) in results/phase1/cloudsim_input/:
 *   settings.csv        server type, power model, PUE, emission factor, time steps
 *   vms_assignment.csv  vm_id, cpu_size_ncu, mem_size_nmu, host_id
 *   vm_cpu_demand.csv   real CPU demand (NCU) of every VM in every 5-minute window
 *
 * Mapping to CloudSim:
 *   1 NCU = MIPS_PER_NCU MIPS, 1 NMU = MB_PER_NMU MB (scaling only, ratios stay the same).
 *   A VM gets exactly its reserved size; its cloudlet uses min(demand / size, 1) of it,
 *   so demand above the size is throttled, like in the Python model.
 *   A host with no VM is treated as switched off (0 W).
 *
 * Outputs in results/phase1/:
 *   cloudsim_energy_hosts.csv, cloudsim_summary.csv
 */
public class Phase1Simulation {
    private static final double MIPS_PER_NCU = 100_000;
    private static final double MB_PER_NMU = 1_000_000;

    private static final Path INPUT = Paths.get("..", "results", "phase1", "cloudsim_input");
    private static final Path OUTPUT = Paths.get("..", "results", "phase1");

    public static void main(String[] args) throws IOException {
        // the cloudlets are meant to run past 24 h, so CloudSim's end-of-run INFO messages are expected noise
        Log.setLevel(Level.WARN);
        Map<String, Double> cfg = readSettings(INPUT.resolve("settings.csv"));
        int numHosts = cfg.get("num_servers").intValue();
        double hostCpu = cfg.get("cpu_capacity_ncu");
        double hostMem = cfg.get("mem_capacity_nmu");
        double idleW = cfg.get("idle_power_w");
        double peakW = cfg.get("peak_power_w");
        double windowSec = cfg.get("window_seconds");
        int windows = cfg.get("windows").intValue();
        double pue = cfg.get("pue");
        double ef = cfg.get("emission_factor");
        double simEnd = windowSec * windows;

        List<double[]> vmRows = readCsvNumbers(INPUT.resolve("vms_assignment.csv"));
        double[][] demand = readDemand(INPUT.resolve("vm_cpu_demand.csv"), vmRows.size());

        CloudSimPlus simulation = new CloudSimPlus();

        List<Host> hosts = new ArrayList<>();
        for (int h = 0; h < numHosts; h++) {
            List<Pe> pes = List.of(new PeSimple(hostCpu * MIPS_PER_NCU));
            Host host = new HostSimple((long) Math.floor(hostMem * MB_PER_NMU), 1_000_000, 10_000_000, pes);
            host.setVmScheduler(new VmSchedulerTimeShared());
            host.setPowerModel(new PowerModelHostSimple(peakW, idleW));
            hosts.add(host);
        }

        // place every VM on the host chosen by the QA optimizer
        Map<Long, Integer> assignment = new HashMap<>();
        for (double[] r : vmRows) {
            assignment.put((long) r[0], (int) r[3]);
        }
        VmAllocationPolicySimple policy = new VmAllocationPolicySimple();
        policy.setFindHostForVmFunction((p, vm) -> Optional.ofNullable(assignment.get(vm.getId())).map(hosts::get));

        DatacenterSimple datacenter = new DatacenterSimple(simulation, hosts, policy);
        datacenter.setSchedulingInterval(windowSec);

        DatacenterBroker broker = new DatacenterBrokerSimple(simulation);
        List<Vm> vms = new ArrayList<>();
        List<Cloudlet> cloudlets = new ArrayList<>();
        for (int i = 0; i < vmRows.size(); i++) {
            double[] r = vmRows.get(i);
            long vmId = (long) r[0];
            double sizeNcu = r[1];
            double vmMips = sizeNcu * MIPS_PER_NCU;

            Vm vm = new VmSimple(vmId, vmMips, 1);
            vm.setRam((long) Math.ceil(r[2] * MB_PER_NMU)).setBw(1000).setSize(1000);
            vm.setCloudletScheduler(new CloudletSchedulerTimeShared());
            vms.add(vm);

            final double[] series = demand[i];
            UtilizationModelDynamic cpuModel = new UtilizationModelDynamic();
            cpuModel.setUtilizationUpdateFunction(um -> {
                int w = (int) Math.min(windows - 1, Math.floor(simulation.clock() / windowSec));
                return Math.min(1.0, series[w] / sizeNcu);
            });
            // long enough to keep running for the whole day
            Cloudlet c = new CloudletSimple((long) (vmMips * simEnd * 10), 1, cpuModel);
            c.setUtilizationModelRam(new UtilizationModelFull());
            c.setUtilizationModelBw(new UtilizationModelDynamic(0.0));
            c.setSizes(1);
            c.setVm(vm);
            cloudlets.add(c);
        }
        broker.submitVmList(vms);
        broker.submitCloudletList(cloudlets);

        // integrate energy exactly: power is constant between two clock ticks
        double[] energyJ = new double[numHosts];
        double[] utilSum = new double[numHosts];
        double[] utilPeak = new double[numHosts];
        int[] vmCount = new int[numHosts];
        double[] lastTime = {0.0};
        int[] samples = {0};
        simulation.addOnClockTickListener(info -> {
            double now = Math.min(info.getTime(), simEnd);
            double dt = now - lastTime[0];
            if (dt <= 0) {
                return;
            }
            for (int h = 0; h < numHosts; h++) {
                Host host = hosts.get(h);
                if (host.getVmList().isEmpty()) {
                    continue;   // switched off
                }
                vmCount[h] = Math.max(vmCount[h], host.getVmList().size());
                double u = Math.min(1.0, host.getCpuPercentUtilization());
                energyJ[h] += host.getPowerModel().getPower(u) * dt;
                utilSum[h] += u * dt;
                utilPeak[h] = Math.max(utilPeak[h], u);
            }
            samples[0]++;
            lastTime[0] = now;
        });

        simulation.terminateAt(simEnd);
        simulation.start();

        report(hosts, vmCount, energyJ, utilSum, utilPeak, lastTime[0], pue, ef, samples[0]);
    }

    private static void report(List<Host> hosts, int[] vmCount, double[] energyJ, double[] utilSum, double[] utilPeak,
                               double simulatedSec, double pue, double ef, int samples) throws IOException {
        StringBuilder csv = new StringBuilder("host_id,active,vms,mean_cpu_util_pct,peak_cpu_util_pct,energy_kwh\n");
        double itKwh = 0;
        int active = 0;
        System.out.println("\n=== CLOUDSIM PHASE 1 RESULT (24 h, QA placement) ===");
        for (int h = 0; h < hosts.size(); h++) {
            boolean on = vmCount[h] > 0;
            double kwh = energyJ[h] / 3_600_000.0;
            double meanUtil = simulatedSec > 0 ? utilSum[h] / simulatedSec * 100 : 0;
            itKwh += kwh;
            if (on) {
                active++;
                System.out.printf(Locale.US, "  Host %2d: VMs=%d, mean CPU=%.1f%%, peak CPU=%.1f%%, energy=%.4f kWh%n",
                        h, vmCount[h], meanUtil, utilPeak[h] * 100, kwh);
            }
            csv.append(String.format(Locale.US, "%d,%b,%d,%.4f,%.4f,%.6f%n",
                    h, on, vmCount[h], meanUtil, utilPeak[h] * 100, kwh));
        }
        double facility = itKwh * pue;
        double co2 = facility * ef;
        System.out.printf(Locale.US, "  Simulated time  : %.0f s (%.1f h), %d clock ticks%n",
                simulatedSec, simulatedSec / 3600, samples);
        System.out.printf(Locale.US, "  Active servers  : %d of %d%n", active, hosts.size());
        System.out.printf(Locale.US, "  IT energy       : %.4f kWh%n", itKwh);
        System.out.printf(Locale.US, "  Facility energy : %.4f kWh (PUE %.2f)%n", facility, pue);
        System.out.printf(Locale.US, "  Carbon          : %.4f kg CO2 (EF %.3f)%n", co2, ef);

        Files.writeString(OUTPUT.resolve("cloudsim_energy_hosts.csv"), csv.toString());
        String summary = String.format(Locale.US,
                "metric,value%nactive_hosts,%d%nit_energy_kwh,%.6f%nfacility_energy_kwh,%.6f%nco2_kg,%.6f%nsimulated_seconds,%.0f%n",
                active, itKwh, facility, co2, simulatedSec);
        Files.writeString(OUTPUT.resolve("cloudsim_summary.csv"), summary);
        System.out.println("  Saved: " + OUTPUT.resolve("cloudsim_summary.csv"));
    }

    private static Map<String, Double> readSettings(Path path) throws IOException {
        Map<String, Double> map = new HashMap<>();
        List<String> lines = Files.readAllLines(path);
        for (String line : lines.subList(1, lines.size())) {
            if (line.isBlank()) {
                continue;
            }
            String[] p = line.split(",");
            map.put(p[0].trim(), Double.parseDouble(p[1].trim()));
        }
        return map;
    }

    private static List<double[]> readCsvNumbers(Path path) throws IOException {
        List<double[]> rows = new ArrayList<>();
        List<String> lines = Files.readAllLines(path);
        for (String line : lines.subList(1, lines.size())) {
            if (line.isBlank()) {
                continue;
            }
            String[] p = line.split(",");
            double[] r = new double[p.length];
            for (int i = 0; i < p.length; i++) {
                r[i] = Double.parseDouble(p[i].trim());
            }
            rows.add(r);
        }
        return rows;
    }

    /** Returns demand[vm][window]; columns follow the same VM order as vms_assignment.csv. */
    private static double[][] readDemand(Path path, int numVms) throws IOException {
        List<double[]> rows = readCsvNumbers(path);
        double[][] d = new double[numVms][rows.size()];
        for (int w = 0; w < rows.size(); w++) {
            for (int v = 0; v < numVms; v++) {
                d[v][w] = rows.get(w)[v + 1];   // column 0 is window_idx
            }
        }
        return d;
    }
}
