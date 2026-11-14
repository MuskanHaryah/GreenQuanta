package greenquanta;

import ch.qos.logback.classic.Level;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicy;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicyBestFit;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicyFirstFit;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicyRoundRobin;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.vm.VmSchedulerSpaceShared;
import org.cloudsimplus.util.Log;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * GreenQuanta Guide 1 step 1.4: VM placement by CloudSim Plus 8.0.0's own allocation policies (placement only, no
 * energy). The built-in Best/Worst-fit rank hosts by free PEs, so CPU is modelled as one PE per quantization unit
 * (written by optimizer/demo/baselines.py): "free PEs" = free CPU units of the safe limit, as in the quantized model.
 *
 *   mvn -q exec:java "-Dsim.main=greenquanta.BaselinePlacement" "-Dexec.args=<in_dir> <out_dir> <policy>"
 *
 * policy: firstfit | bestfit | worstfit | roundrobin   built-in policy, VMs submitted in input order
 *         ffd | bfd                                    textbook first-fit (index order) / built-in BestFit, VMs sorted
 *                                                      by CPU desc, then memory desc, then id
 *         pabfd                                        Power-Aware Best-Fit Decreasing (Beloglazov & Buyya 2012, Alg. 1):
 *                                                      sorted as ffd, each VM to the suitable host with the smallest
 *                                                      power increase (Phase2Simulation.specPower; a host with no VM is
 *                                                      off, 0 W). Not built into CloudSim Plus 8.
 * Inputs (in_dir): servers.csv, vms_assignment.csv (host_id ignored), baseline_hosts.csv (host_id,pes,ram_mb),
 *                  baseline_vms.csv (vm_id,pes,ram_mb)
 * Outputs (out_dir): vms_assignment.csv (placed VMs, same columns), failed_vms.csv (vm_id),
 *                   submitted_order.csv (vm_id in the order given to the broker)
 */
public class BaselinePlacement {
    private static final double MIPS_PER_PE = 250;   // any value: hosts and VMs use the same PE size

    public static void main(String[] args) throws IOException {
        Path in = Paths.get(args[0]);
        Path out = Paths.get(args[1]);
        String policyName = args[2];
        Log.setLevel(Level.WARN);
        Files.createDirectories(out);

        List<String[]> serverRows = Phase2Simulation.readCsvText(in.resolve("servers.csv"));
        List<double[]> vmRows = Phase2Simulation.readCsvNumbers(in.resolve("vms_assignment.csv"));
        List<double[]> hostModel = Phase2Simulation.readCsvNumbers(in.resolve("baseline_hosts.csv"));
        Map<Long, double[]> vmModel = new HashMap<>();
        for (double[] r : Phase2Simulation.readCsvNumbers(in.resolve("baseline_vms.csv"))) {
            vmModel.put((long) r[0], r);
        }

        CloudSimPlus simulation = new CloudSimPlus();
        List<Host> hosts = new ArrayList<>();
        double[] capacity = new double[serverRows.size()];
        double[][] curves = new double[serverRows.size()][11];
        for (int h = 0; h < serverRows.size(); h++) {
            String[] r = serverRows.get(h);
            capacity[h] = Double.parseDouble(r[2]);
            for (int k = 0; k < 11; k++) {
                curves[h][k] = Double.parseDouble(r[4 + k]);
            }
            List<Pe> pes = new ArrayList<>();
            for (int p = 0; p < (int) hostModel.get(h)[1]; p++) {
                pes.add(new PeSimple(MIPS_PER_PE));
            }
            Host host = new HostSimple((long) hostModel.get(h)[2], 1_000_000, 10_000_000, pes);
            host.setId(h);
            host.setVmScheduler(new VmSchedulerSpaceShared());
            hosts.add(host);
        }

        Map<Long, Double> cpuNcu = new HashMap<>();
        List<Vm> vms = new ArrayList<>();
        for (double[] r : vmRows) {
            long id = (long) r[0];
            cpuNcu.put(id, r[1]);
            double[] m = vmModel.get(id);
            Vm vm = new VmSimple(id, MIPS_PER_PE, (long) m[1]);
            vm.setRam((long) m[2]).setBw(1000).setSize(1000);
            vms.add(vm);
        }
        boolean sorted = policyName.equals("ffd") || policyName.equals("bfd") || policyName.equals("pabfd");
        if (sorted) {
            Map<Long, Double> mem = new HashMap<>();
            vmRows.forEach(r -> mem.put((long) r[0], r[2]));
            vms.sort(Comparator.comparing((Vm v) -> -cpuNcu.get(v.getId()))
                    .thenComparing(v -> -mem.get(v.getId())).thenComparingLong(Vm::getId));
        }

        VmAllocationPolicy policy = switch (policyName) {
            case "firstfit" -> new VmAllocationPolicyFirstFit();
            case "bestfit", "bfd" -> new VmAllocationPolicyBestFit();
            case "worstfit" -> new VmAllocationPolicySimple();
            case "roundrobin" -> new VmAllocationPolicyRoundRobin();
            case "ffd" -> new VmAllocationPolicySimple(
                    (p, vm) -> p.getHostList().stream().filter(h -> h.isSuitableForVm(vm)).findFirst());
            case "pabfd" -> new VmAllocationPolicySimple((p, vm) -> {
                Host best = null;
                double bestIncrease = Double.POSITIVE_INFINITY;
                for (Host h : p.getHostList()) {
                    if (!h.isSuitableForVm(vm)) {
                        continue;
                    }
                    int i = (int) h.getId();
                    double before = h.getVmList().stream().mapToDouble(v -> cpuNcu.get(v.getId())).sum();
                    double after = before + cpuNcu.get(vm.getId());
                    double powerBefore = h.getVmList().isEmpty() ? 0 : Phase2Simulation.specPower(curves[i], before / capacity[i]);
                    double increase = Phase2Simulation.specPower(curves[i], after / capacity[i]) - powerBefore;
                    if (increase < bestIncrease) {
                        bestIncrease = increase;
                        best = h;
                    }
                }
                return Optional.ofNullable(best);
            });
            default -> throw new IllegalArgumentException("unknown policy " + policyName);
        };

        new DatacenterSimple(simulation, hosts, policy);
        DatacenterBrokerSimple broker = new DatacenterBrokerSimple(simulation);
        broker.setVmComparator(null);   // keep the submission order
        Map<Long, Long> placed = new HashMap<>();
        vms.forEach(vm -> vm.addOnHostAllocationListener(e -> placed.put(e.getVm().getId(), e.getHost().getId())));
        broker.submitVmList(vms);
        simulation.start();

        StringBuilder assignment = new StringBuilder("vm_id,cpu_size_ncu,mem_size_nmu,host_id\n");
        StringBuilder failed = new StringBuilder("vm_id\n");
        for (double[] r : vmRows) {
            Long host = placed.get((long) r[0]);
            if (host == null) {
                failed.append((long) r[0]).append('\n');
            } else {
                assignment.append(String.format(Locale.US, "%d,%s,%s,%d%n", (long) r[0], r[1], r[2], host));
            }
        }
        Files.writeString(out.resolve("vms_assignment.csv"), assignment.toString());
        Files.writeString(out.resolve("failed_vms.csv"), failed.toString());
        StringBuilder order = new StringBuilder("vm_id\n");
        vms.forEach(vm -> order.append(vm.getId()).append('\n'));
        Files.writeString(out.resolve("submitted_order.csv"), order.toString());
        System.out.printf(Locale.US, "BASELINE %s: placed %d of %d VMs on %d hosts%n", policyName, placed.size(),
                vmRows.size(), placed.values().stream().distinct().count());
    }
}
