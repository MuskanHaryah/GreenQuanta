package greenquanta;

import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.builders.tables.CloudletsTableBuilder;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.util.ArrayList;
import java.util.List;

public class BasicSimulation {
    private static final int HOSTS = 2;
    private static final int HOST_PES = 8;
    private static final int VMS = 4;
    private static final int VM_PES = 2;
    private static final int CLOUDLETS = 8;

    public static void main(String[] args) {
        CloudSimPlus simulation = new CloudSimPlus();

        List<Host> hostList = new ArrayList<>();
        for (int i = 0; i < HOSTS; i++) {
            List<Pe> peList = new ArrayList<>();
            for (int p = 0; p < HOST_PES; p++) {
                peList.add(new PeSimple(1000));
            }
            hostList.add(new HostSimple(16384, 10000, 1000000, peList));
        }
        new DatacenterSimple(simulation, hostList);

        DatacenterBroker broker = new DatacenterBrokerSimple(simulation);

        List<Vm> vmList = new ArrayList<>();
        for (int i = 0; i < VMS; i++) {
            Vm vm = new VmSimple(1000, VM_PES);
            vm.setRam(2048).setBw(1000).setSize(10000);
            vmList.add(vm);
        }

        List<Cloudlet> cloudletList = new ArrayList<>();
        UtilizationModelDynamic utilization = new UtilizationModelDynamic(0.5);
        for (int i = 0; i < CLOUDLETS; i++) {
            Cloudlet cloudlet = new CloudletSimple(10000, VM_PES, utilization);
            cloudlet.setSizes(1024);
            cloudletList.add(cloudlet);
        }

        broker.submitVmList(vmList);
        broker.submitCloudletList(cloudletList);

        simulation.start();

        new CloudletsTableBuilder(broker.getCloudletFinishedList()).build();

        printSummary(hostList, vmList, broker.getCloudletFinishedList());
    }

    private static void printSummary(List<Host> hostList, List<Vm> vmList, List<Cloudlet> finished) {
        System.out.println("\n================ SIMULATION SUMMARY ================");
        System.out.println("Total hosts     : " + hostList.size());
        System.out.println("Total VMs       : " + vmList.size());
        System.out.println("Total cloudlets : " + finished.size());

        System.out.println("\n--- Hosts ---");
        long totalHostCores = 0;
        long totalHostRam = 0;
        for (Host h : hostList) {
            long cores = h.getPesNumber();
            long ram = h.getRam().getCapacity();
            totalHostCores += cores;
            totalHostRam += ram;
            System.out.printf("Host %d: CPU cores = %d (1000 MIPS each), RAM = %d MB%n",
                h.getId(), cores, ram);
        }
        System.out.printf("Total host capacity: CPU cores = %d, RAM = %d MB%n",
            totalHostCores, totalHostRam);

        System.out.println("\n--- VMs ---");
        long totalVmCores = 0;
        long totalVmRam = 0;
        for (Vm vm : vmList) {
            long cores = vm.getPesNumber();
            long ram = vm.getRam().getCapacity();
            totalVmCores += cores;
            totalVmRam += ram;
            System.out.printf("VM %d: CPU cores = %d, RAM = %d MB, placed on Host %d%n",
                vm.getId(), cores, ram, vm.getHost().getId());
        }
        System.out.printf("Total VM demand: CPU cores = %d, RAM = %d MB%n",
            totalVmCores, totalVmRam);

        System.out.println("\n--- Cloudlets ---");
        System.out.println("Each cloudlet: length = 10000 MI, CPU cores = " + VM_PES
            + ", CPU usage = 50%");
        System.out.println("====================================================");
    }
}
