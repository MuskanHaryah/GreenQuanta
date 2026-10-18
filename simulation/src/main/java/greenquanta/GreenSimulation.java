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
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class GreenSimulation {
    private static final int HOSTS = 2;
    private static final int HOST_PES = 8;
    private static final int VMS = 4;
    private static final int VM_PES = 2;
    private static final int CLOUDLETS = 8;

    // Power model (watts): idle host 100 W, fully busy host 250 W
    private static final double MAX_POWER = 250;
    private static final double STATIC_POWER = 100;

    public static void main(String[] args) throws IOException {
        String mode = args.length > 0 ? args[0] : "default";
        System.out.println("Mode: " + mode);

        CloudSimPlus simulation = new CloudSimPlus();
        List<Host> hostList = createHosts();

        VmAllocationPolicySimple policy = new VmAllocationPolicySimple();
        if (mode.equals("qa")) {
            Map<Long, Integer> assignment =
                readAssignment(Paths.get("..", "results", "assignment_qa.csv"));
            policy.setFindHostForVmFunction((p, vm) -> {
                Integer hostIndex = assignment.get(vm.getId());
                return hostIndex == null ? Optional.empty() : Optional.of(hostList.get(hostIndex));
            });
        }

        DatacenterSimple datacenter = new DatacenterSimple(simulation, hostList, policy);
        datacenter.setSchedulingInterval(1);

        DatacenterBroker broker = new DatacenterBrokerSimple(simulation);
        broker.submitVmList(createVms());
        broker.submitCloudletList(createCloudlets());

        simulation.start();

        report(mode, hostList, broker.getCloudletFinishedList());
    }

    private static List<Host> createHosts() {
        List<Host> list = new ArrayList<>();
        for (int i = 0; i < HOSTS; i++) {
            List<Pe> peList = new ArrayList<>();
            for (int p = 0; p < HOST_PES; p++) {
                peList.add(new PeSimple(1000));
            }
            Host host = new HostSimple(16384, 10000, 1000000, peList);
            host.setPowerModel(new PowerModelHostSimple(MAX_POWER, STATIC_POWER));
            host.enableUtilizationStats();
            list.add(host);
        }
        return list;
    }

    private static List<Vm> createVms() {
        List<Vm> list = new ArrayList<>();
        for (int i = 0; i < VMS; i++) {
            Vm vm = new VmSimple(i, 1000, VM_PES);
            vm.setRam(2048).setBw(1000).setSize(10000);
            list.add(vm);
        }
        return list;
    }

    private static List<Cloudlet> createCloudlets() {
        List<Cloudlet> list = new ArrayList<>();
        UtilizationModelDynamic utilization = new UtilizationModelDynamic(0.5);
        for (int i = 0; i < CLOUDLETS; i++) {
            Cloudlet cloudlet = new CloudletSimple(10000, VM_PES, utilization);
            cloudlet.setSizes(1024);
            list.add(cloudlet);
        }
        return list;
    }

    private static Map<Long, Integer> readAssignment(Path path) throws IOException {
        Map<Long, Integer> map = new HashMap<>();
        List<String> lines = Files.readAllLines(path);
        for (String line : lines.subList(1, lines.size())) {
            if (line.isBlank()) {
                continue;
            }
            String[] parts = line.split(",");
            map.put(Long.parseLong(parts[0].trim()), Integer.parseInt(parts[1].trim()));
        }
        System.out.println("QA assignment loaded: " + map);
        return map;
    }

    private static void report(String mode, List<Host> hostList, List<Cloudlet> finished)
            throws IOException {
        Set<Long> activeHostIds = new HashSet<>();
        for (Cloudlet c : finished) {
            activeHostIds.add(c.getVm().getHost().getId());
        }

        StringBuilder csv = new StringBuilder("host_id,active,mean_cpu_util,mean_power_w,energy_wh\n");
        double totalEnergy = 0;
        int activeCount = 0;

        System.out.println("\n=== CLOUDSIM ENERGY RESULT (" + mode + ") ===");
        for (Host h : hostList) {
            boolean active = activeHostIds.contains(h.getId());
            double util = h.getCpuUtilizationStats().getMean();
            // An empty host is assumed to be switched off (0 W)
            double power = active ? h.getPowerModel().getPower(util) : 0.0;
            double energyWh = power * h.getTotalUpTime() / 3600.0;

            if (active) {
                activeCount++;
            }
            totalEnergy += energyWh;

            System.out.printf(Locale.US,
                "  Host %d: active=%b, CPU util=%.1f%%, power=%.1f W, energy=%.4f Wh%n",
                h.getId(), active, util * 100, power, energyWh);
            csv.append(String.format(Locale.US, "%d,%b,%.4f,%.2f,%.6f%n",
                h.getId(), active, util, power, energyWh));
        }

        System.out.println("  Active hosts: " + activeCount);
        System.out.printf(Locale.US, "  Total energy: %.4f Wh%n", totalEnergy);

        Path out = Paths.get("..", "results", "cloudsim_" + mode + ".csv");
        Files.writeString(out, csv.toString());
        System.out.println("  Saved: " + out);
    }
}