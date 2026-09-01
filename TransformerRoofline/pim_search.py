import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from roofline import TransformerRoofline, Hardware
from copy import deepcopy

class PIMv1DSE:
    def __init__(self, comb_hard, batchsize, model, ratio_p, max_token, XPU_baseline, list_TFLOPS, list_BW_TBs, list_PIM_Capacity):
        self.roofline = TransformerRoofline('hardware_models.json')
        self.comb_hard = comb_hard
        self.comb_hard_ = deepcopy(comb_hard)
        self.batchsize = batchsize
        self.model = model
        self.ratio_p = ratio_p
        self.max_token = max_token
        self.XPU = XPU_baseline
        self.list_TFLOPS = list_TFLOPS
        self.list_BW_TBs = list_BW_TBs
        self.list_PIM_Capacity = list_PIM_Capacity
        self.required_token_per_second = 0

    def cal_baseline(self):
        """
        Calculate GPU baseline performance
        """
        roofline = self.roofline
        XPU = self.XPU
        
        comb_hard = {
            "Name":"8"+str(XPU), 
            "PRF":{"hard":XPU, "quant":8},
            "GNR-PRJ":{"hard":XPU, "quant":8}, 
            "GNR-ACT":{"hard":XPU, "quant":8}, 
            "Assign":"PRF+GNR-PRJ+GNR-ACT"
        }

        comb_hard["Offcard_Type"] = []
        comb_hard["PRF"]["hard"] = roofline.hardwares[XPU]
        comb_hard["GNR-PRJ"]["hard"] = roofline.hardwares[XPU]
        comb_hard["GNR-ACT"]["hard"] = roofline.hardwares[XPU]

        # Add baseline hardware
        roofline.hardwares[comb_hard["Name"]] = Hardware(comb_hard, is_combination=True)

        # Calculate the largest batchsize; constraint = ["All", "FPRT", "Latency", "None"]
        max_batchsize = roofline.Cal_Max_Batchsize(ratiop=self.ratio_p, maxtoken=self.max_token,\
             model=self.model, hardware=comb_hard["Name"], constraint="All") 

        # Calculate other metrics
        roofline.Cal_Metric(Ratio_P=[self.ratio_p], Batchsize=[max_batchsize], Max_Token=[self.max_token], \
            Model=[self.model], Hardware=[comb_hard["Name"]], Sparse=["default"])

        self.baseline_FPRT = roofline.FPRT
        self.baseline_Latency = roofline.Latency
        # self.baseline_Query_per_Sec = roofline.Query_per_Sec
        self.baseline_Token_per_Sec = roofline.Token_per_Sec
        # self.baseline_SLO = roofline.SLO
        self.baseline_Price_per_Token = roofline.Price_per_Token
        self.baseline_Max_Batchsize = max_batchsize

    def cal_metric(self, ratiop, batchsize, maxtoken, model, hardware):
        all_data = self.roofline.Compute_Timebreakdown(Ratio_P=[ratiop], Batchsize=[batchsize], Max_Token=[maxtoken], Model=[model],\
             Hardware=[hardware], Sparse=["default"], Pipeline_Stage=1)
        maxtoken_info = all_data[hardware][model][ratiop][batchsize]["default"][maxtoken]
        Pipeline_Stage = 1
        Throughput_Ratio = Pipeline_Stage
        FPRT = (maxtoken_info["Timebreakdown"]["PRF"] + maxtoken_info["Timebreakdown"]["OFFCARD"]["PRF"]["Total"])
        Latency = Throughput_Ratio * (maxtoken_info["Timebreakdown"]["Total"] - maxtoken_info["Timebreakdown"]["PRF"] - maxtoken_info["Timebreakdown"]["OFFCARD"]["PRF"]["Total"]) / (maxtoken * (1-ratiop))
        # Token_per_Sec = maxtoken_info["batchsize"] * maxtoken * Throughput_Ratio / (maxtoken_info["Timebreakdown"]["Total"])
        return FPRT, Latency

    def cal_baseline_serving_system_perf(self):
        """
        Calculate GPU serving system baseline performance
        """
        roofline = self.roofline
        XPU = self.XPU
        
        comb_hard = {
            "Name":"8"+str(XPU), 
            "PRF":{"hard":XPU, "quant":8},
            "GNR-PRJ":{"hard":XPU, "quant":8}, 
            "GNR-ACT":{"hard":XPU, "quant":8}, 
            "Assign":"PRF+GNR-PRJ+GNR-ACT"
        }

        comb_hard["Offcard_Type"] = []
        comb_hard["PRF"]["hard"] = roofline.hardwares[XPU]
        comb_hard["GNR-PRJ"]["hard"] = roofline.hardwares[XPU]
        comb_hard["GNR-ACT"]["hard"] = roofline.hardwares[XPU]

        # Add new hardware
        roofline.hardwares[comb_hard["Name"]] = Hardware(comb_hard, is_combination=True)

        # Calculate batchsize under different constraint-bound
        max_batchsize_capacity_bound, max_batchsize_FPRT_bound, max_batchsize_latency_bound = \
        roofline.Get_Max_Batch(hardware=comb_hard["Name"],model=self.model,ratiop=self.ratio_p,maxtoken=self.max_token,\
            sparse="default",pipeline=1,max_FPRT=3,min_token_per_sec=20,first_col=True,output=True)

        # In serving system, the latency-bound batchsize should not larger than cap-bound results
        if max_batchsize_latency_bound > max_batchsize_capacity_bound:
            max_batchsize_latency_bound = max_batchsize_capacity_bound

        FPRT, _ = self.cal_metric(ratiop=self.ratio_p, batchsize=max_batchsize_FPRT_bound, maxtoken=self.max_token,\
            model=self.model, hardware=comb_hard["Name"])
        time_PRF = FPRT

        _, Latency = self.cal_metric(ratiop=self.ratio_p, batchsize=max_batchsize_latency_bound, maxtoken=self.max_token,\
            model=self.model, hardware=comb_hard["Name"])
        time_GNR = Latency * (self.max_token * (1 - self.ratio_p))

        # The batchsize is Mem-Bound or Latency-Bound
        if max_batchsize_latency_bound == max_batchsize_capacity_bound:
            status_bs = "Mem-Bound"
        elif max_batchsize_latency_bound < max_batchsize_capacity_bound:
            status_bs = "Lat-Bound"

        # Calculate how many device needed in PRF phase
        m = int(np.ceil(max_batchsize_latency_bound / max_batchsize_FPRT_bound))

        # Calculate how many device needed in GNR phase
        n = int(np.ceil(time_GNR / time_PRF))

        # Calculate total throught (PRF+GNR) under "m x PRF + n x GNR" system 
        perf = max_batchsize_latency_bound * self.max_token / time_PRF 
        # perf per device; will be discarded
        perf_per_cost = perf / (m + n)

        # Calculte system TCO cost and normalized cost perf token
        roofline.cost_model.init_hardware(roofline.hardwares[comb_hard["Name"]])
        price_per_token = roofline.cost_model.cal_system_cost(perf, m, n)

        print("\033[1;31m******* Serving System GPU Baseline Config *******\033[0m")
        print("\033[1;31m%-8s%-8s%-8s%-2s%-8s%-8s%-8s%-2s%-8s%-10s%-2s%-12s%-12s\033[0m"%(
            "PRF-num", "PRF-bs", "PRF-time", "|",
            "GNR-num", "GNR-bs", "GNR-time", "|",
            "Max-bs", "Status-bs", "|", 
            "Sys-Perf", "Cost/Token"))
        print("{:<8}{:<8}{:<8.2f}{:<2}{:<8}{:<8}{:<8.2f}{:<2}{:<8}{:<10}{:<2}{:<12.2f}{:<12.3e}".format(
            m, max_batchsize_FPRT_bound, time_PRF, "|",
            n, max_batchsize_latency_bound, time_GNR, "|",
            max_batchsize_capacity_bound, status_bs, "|",
            perf, price_per_token
        ))

        self.baseline_perf = perf
        self.baseline_perf_per_cost = perf_per_cost
        self.baseline_price_per_token = price_per_token

    def cal_serving_system_perf(self):
        """
        Calculate XPU+PIM serving system performance
        """

        roofline = self.roofline
        comb_hard = self.comb_hard
        XPU = self.XPU

        self.perf = []
        self.perf_per_cost = []
        self.price_per_token = []
        self.improve = []
        self.num_PIM_chip = []
        self.PIM_device_price = []
        self.num_of_mem_layer = []

        assert comb_hard["Assign"] == "PRF+GNR-PRJ, GNR-ACT", "Only support [PRF+GP,GA] assignment!"
        for tflops in self.list_TFLOPS:
            for cap in self.list_PIM_Capacity:
                roofline.hardwares["PIMv1"].MM_TFLOPS = tflops
                roofline.hardwares["PIMv1"].MM_BW_TBs = tflops
                roofline.hardwares["PIMv1"].MM_GP_TFLOPS = tflops
                roofline.hardwares["PIMv1"].MM_GP_BW_TBs = tflops
                roofline.hardwares["PIMv1"].MV_TFLOPS = tflops
                roofline.hardwares["PIMv1"].MV_BW_TBs = tflops 
                roofline.hardwares["PIMv1"].Capacity = cap
                roofline.hardwares["PIMv1"].price = 0
                comb_hard["Offcard_Type"] = ["Acti"]
                comb_hard["PRF"]["hard"] = roofline.hardwares[XPU]
                comb_hard["GNR-PRJ"]["hard"] = roofline.hardwares[XPU]
                comb_hard["GNR-ACT"]["hard"] = roofline.hardwares["PIMv1"]

                # Add new hardware
                roofline.hardwares[comb_hard["Name"]] = Hardware(comb_hard, is_combination=True)

                # Calculate batchsize under different constraint-bound
                max_batchsize_capacity_bound, max_batchsize_FPRT_bound, max_batchsize_latency_bound = \
                roofline.Get_Max_Batch(hardware=comb_hard["Name"],model=self.model,ratiop=self.ratio_p,maxtoken=self.max_token,\
                    sparse="default",pipeline=1,max_FPRT=3,min_token_per_sec=20,first_col=True,output=True)

                # In serving system, the latency-bound batchsize should not larger than cap-bound results
                if max_batchsize_latency_bound > max_batchsize_capacity_bound:
                    max_batchsize_latency_bound = max_batchsize_capacity_bound

                FPRT, _ = self.cal_metric(ratiop=self.ratio_p, batchsize=max_batchsize_FPRT_bound, maxtoken=self.max_token,\
                    model=self.model, hardware=comb_hard["Name"])
                time_PRF = FPRT

                _, Latency = self.cal_metric(ratiop=self.ratio_p, batchsize=max_batchsize_latency_bound, maxtoken=self.max_token,\
                    model=self.model, hardware=comb_hard["Name"])
                time_GNR = Latency * (self.max_token * (1 - self.ratio_p))

                # The batchsize is Mem-Bound or Latency-Bound
                if max_batchsize_latency_bound == max_batchsize_capacity_bound:
                    status_bs = "Mem-Bound"
                elif max_batchsize_latency_bound < max_batchsize_capacity_bound:
                    status_bs = "Lat-Bound"

                # Calculate how many device needed in PRF phase
                m = int(np.ceil(max_batchsize_latency_bound / max_batchsize_FPRT_bound))

                # Calculate how many device needed in GNR phase
                n = int(np.ceil(time_GNR / time_PRF))

                # Calculate total throught (PRF+GNR) under "m x PRF + n x GNR" system 
                perf = max_batchsize_latency_bound * self.max_token / time_PRF 
                # perf per device; will be discarded
                perf_per_cost = perf / (m + n)

                # Calculte system TCO cost and normalized cost perf token
                roofline.cost_model.init_hardware(roofline.hardwares[comb_hard["Name"]])
                # [TODO] Harecode tradeoff, data from 芯盟
                cap_per_layer = -0.36 * tflops/32 + 24.028
                num_of_mem_layer = int(np.ceil( (cap/32.0) / cap_per_layer ))
                
                if self.required_token_per_second == 0:
                    roofline.cost_model.required_token_per_second = 99000000
                else:
                    roofline.cost_model.required_token_per_second = self.required_token_per_second
                price_per_token = roofline.cost_model.cal_system_cost(perf, m, n, num_of_mem_layer, "PRF+GNR-PRJ, GNR-ACT")

                num_PIM_chip = roofline.cost_model.PIM_device * 32
                self.PIM_device_price.append(roofline.cost_model.PIM_device_price)

                print("\033[1;31m******* Serving System XPU+PIM Config *******\033[0m")
                print("\033[1;31m%-8s%-8s%-8s%-2s%-8s%-8s%-8s%-2s%-8s%-10s%-2s%-12s%-12s%-6s\033[0m"%(
                    "PRF-num", "PRF-bs", "PRF-time", "|",
                    "GNR-num", "GNR-bs", "GNR-time", "|",
                    "Max-bs", "Status-bs", "|", 
                    "Sys-Perf", "Cost/Token", "Impr"))
                print("{:<8}{:<8}{:<8.2f}{:<2}{:<8}{:<8}{:<8.2f}{:<2}{:<8}{:<10}{:<2}{:<12.0f}{:<12.3e}{:<6.2f}".format(
                    m, max_batchsize_FPRT_bound, time_PRF, "|",
                    n, max_batchsize_latency_bound, time_GNR, "|",
                    max_batchsize_capacity_bound, status_bs, "|",
                    perf, price_per_token, self.baseline_price_per_token / price_per_token
                ))
                print("\033[1;31m%-7s%-5s%-6s%-10s%-2s%-12s%-14s%-13s%-12s%-12s%-2s%-8s\033[0m"%(
                    "TFLOPS", "BW", "Cap", "Mem-Layer", "|",
                    "Req-Token/s", "Num(PIM-Chip)", "RMB(PIMChip)", "Num(PIMSys)", "RMB(PIMSys)", "|",
                    "Wafer-Slot"))
                print("{:<7}{:<5}{:<6}{:<10}{:<2}{:<12.2e}{:<14}{:<13}{:<12}{:<12}{:<2}{:<8}".format(
                    tflops, tflops, cap, int(num_of_mem_layer), "|",
                    roofline.cost_model.required_token_per_second, int(np.ceil(num_PIM_chip)), int(roofline.cost_model.PIM_device_price/32), int(np.ceil(roofline.cost_model.PIM_device)), int(roofline.cost_model.PIM_device_price), "|",
                    int(np.ceil(num_PIM_chip/66)), "|"
                ))

                self.perf.append(perf)
                self.perf_per_cost.append(perf_per_cost)
                self.price_per_token.append(price_per_token)
                self.improve.append(self.baseline_price_per_token / price_per_token)
                self.num_PIM_chip.append(num_PIM_chip)
                self.num_of_mem_layer.append(num_of_mem_layer)

    def cal_THRIVE_serving_system_perf(self):
        """
        Calculate THRIVE++ serving system performance
        """

        roofline = self.roofline
        comb_hard = self.comb_hard
        XPU = self.XPU

        self.perf = []
        self.perf_per_cost = []
        self.price_per_token = []
        self.improve = []
        self.num_THRIVE_chip = []
        self.PIM_device_price = []
        self.num_of_mem_layer = []

        assert comb_hard["Assign"] == "PRF+GNR-PRJ+GNR-ACT", "Only support THRIVE++ [PRF+GP+GA] assignment!"

        for bw in self.list_BW_TBs: 
            for cap in self.list_PIM_Capacity:
                roofline.hardwares["THRIVE++"].MM_BW_TBs = bw
                roofline.hardwares["THRIVE++"].MM_GP_BW_TBs = bw
                roofline.hardwares["THRIVE++"].MV_BW_TBs = bw 
                roofline.hardwares["THRIVE++"].Capacity = cap
                roofline.hardwares["THRIVE++"].price = 0
                comb_hard["Offcard_Type"] = []
                comb_hard["PRF"]["hard"] = roofline.hardwares["THRIVE++"]
                comb_hard["GNR-PRJ"]["hard"] = roofline.hardwares["THRIVE++"]
                comb_hard["GNR-ACT"]["hard"] = roofline.hardwares["THRIVE++"]

                # Add new hardware
                roofline.hardwares[comb_hard["Name"]] = Hardware(comb_hard, is_combination=True)

                # Calculate batchsize under different constraint-bound
                max_batchsize_capacity_bound, max_batchsize_FPRT_bound, max_batchsize_latency_bound = \
                roofline.Get_Max_Batch(hardware=comb_hard["Name"],model=self.model,ratiop=self.ratio_p,maxtoken=self.max_token,\
                    sparse="default",pipeline=1,max_FPRT=3,min_token_per_sec=20,first_col=True,output=True)

                # In serving system, the latency-bound batchsize should not larger than cap-bound results
                if max_batchsize_latency_bound > max_batchsize_capacity_bound:
                    max_batchsize_latency_bound = max_batchsize_capacity_bound

                FPRT, _ = self.cal_metric(ratiop=self.ratio_p, batchsize=max_batchsize_FPRT_bound, maxtoken=self.max_token,\
                    model=self.model, hardware=comb_hard["Name"])
                time_PRF = FPRT

                _, Latency = self.cal_metric(ratiop=self.ratio_p, batchsize=max_batchsize_latency_bound, maxtoken=self.max_token,\
                    model=self.model, hardware=comb_hard["Name"])
                time_GNR = Latency * (self.max_token * (1 - self.ratio_p))

                # The batchsize is Mem-Bound or Latency-Bound
                if max_batchsize_latency_bound == max_batchsize_capacity_bound:
                    status_bs = "Mem-Bound"
                elif max_batchsize_latency_bound < max_batchsize_capacity_bound:
                    status_bs = "Lat-Bound"

                # Calculate how many device needed in PRF phase
                m = int(np.ceil(max_batchsize_latency_bound / max_batchsize_FPRT_bound))

                # Calculate how many device needed in GNR phase
                n = int(np.ceil(time_GNR / time_PRF))

                # Calculate total throught (PRF+GNR) under "m x PRF + n x GNR" system 
                perf = max_batchsize_latency_bound * self.max_token / time_PRF 
                # perf per device; will be discarded
                perf_per_cost = perf / (m + n)

                # Calculte system TCO cost and normalized cost perf token
                roofline.cost_model.init_hardware(roofline.hardwares[comb_hard["Name"]])
                # [TODO] Harecode tradeoff, data from 芯盟
                cap_per_layer = -0.36 * bw/32 + 24.028
                num_of_mem_layer = int(np.ceil( cap / cap_per_layer ))
                
                if self.required_token_per_second == 0:
                    roofline.cost_model.required_token_per_second = 99000000
                else:
                    roofline.cost_model.required_token_per_second = self.required_token_per_second
                price_per_token = roofline.cost_model.cal_system_cost(perf, m, n, num_of_mem_layer, "PRF+GNR-PRJ+GNR-ACT")

                num_THRIVE_chip = roofline.cost_model.XPU_device * roofline.cost_model.XPU.num

                print("\033[1;31m******* Serving System XPU+PIM Config *******\033[0m")
                print("\033[1;31m%-8s%-8s%-8s%-2s%-8s%-8s%-8s%-2s%-8s%-10s%-2s%-12s%-12s%-6s\033[0m"%(
                    "PRF-num", "PRF-bs", "PRF-time", "|",
                    "GNR-num", "GNR-bs", "GNR-time", "|",
                    "Max-bs", "Status-bs", "|", 
                    "Sys-Perf", "Cost/Token", "Impr"))
                print("{:<8}{:<8}{:<8.2f}{:<2}{:<8}{:<8}{:<8.2f}{:<2}{:<8}{:<10}{:<2}{:<12.0f}{:<12.3e}{:<6.2f}".format(
                    m, max_batchsize_FPRT_bound, time_PRF, "|",
                    n, max_batchsize_latency_bound, time_GNR, "|",
                    max_batchsize_capacity_bound, status_bs, "|",
                    perf, price_per_token, self.baseline_price_per_token / price_per_token
                ))

                tflops = roofline.hardwares["THRIVE++"].MM_TFLOPS
                print("\033[1;31m%-7s%-5s%-6s%-10s%-2s%-12s%-14s%-13s%-12s%-12s%-2s%-8s\033[0m"%(
                    "TFLOPS", "BW", "Cap", "Mem-Layer", "|",
                    "Req-Token/s", "Num(T++Chip)", "RMB(T++Chip)", "Num(T++Sys)", "RMB(T++Sys)", "|",
                    "Wafer-Slot"))
                print("{:<7}{:<5.1f}{:<6}{:<10}{:<2}{:<12.2e}{:<13}{:<13}{:<12}{:<12}{:<2}{:<8}".format(
                    tflops, bw, cap, int(num_of_mem_layer), "|",
                    roofline.cost_model.required_token_per_second, int(np.ceil(num_THRIVE_chip)), int(roofline.cost_model.XPU.price), int(np.ceil(roofline.cost_model.XPU_device)), int(roofline.cost_model.XPU_device_price), "|",
                    int(np.ceil(num_THRIVE_chip/66)), "|"
                ))

                self.perf.append(perf)
                self.perf_per_cost.append(perf_per_cost)
                self.price_per_token.append(price_per_token)
                self.improve.append(self.baseline_price_per_token / price_per_token)
                self.num_THRIVE_chip.append(num_THRIVE_chip)
                self.num_of_mem_layer.append(num_of_mem_layer)

    def cal_THRIVE_serving_system_perf_v2(self):
        """
        Calculate THRIVE++ serving system performance
        """

        roofline = self.roofline
        comb_hard = self.comb_hard
        XPU = self.XPU

        self.perf = []
        self.perf_per_cost = []
        self.price_per_token = []
        self.improve = []
        self.num_THRIVE_chip = []
        self.PIM_device_price = []
        self.num_of_mem_layer = []

        assert comb_hard["Assign"] == "PRF+GNR-PRJ+GNR-ACT", "Only support THRIVE++ [PRF+GP+GA] assignment!"

        cap = self.list_PIM_Capacity[0]
        for bw in self.list_BW_TBs: 
            for tflops in self.list_TFLOPS:
                roofline.hardwares["THRIVE++"].MM_TFLOPS = tflops
                roofline.hardwares["THRIVE++"].MM_GP_TFLOPS = tflops
                roofline.hardwares["THRIVE++"].MV_TFLOPS = tflops

                roofline.hardwares["THRIVE++"].MM_BW_TBs = bw
                roofline.hardwares["THRIVE++"].MM_GP_BW_TBs = bw
                roofline.hardwares["THRIVE++"].MV_BW_TBs = bw 
                roofline.hardwares["THRIVE++"].Capacity = cap
                roofline.hardwares["THRIVE++"].price = 0
                comb_hard["Offcard_Type"] = []
                comb_hard["PRF"]["hard"] = roofline.hardwares["THRIVE++"]
                comb_hard["GNR-PRJ"]["hard"] = roofline.hardwares["THRIVE++"]
                comb_hard["GNR-ACT"]["hard"] = roofline.hardwares["THRIVE++"]

                # Add new hardware
                roofline.hardwares[comb_hard["Name"]] = Hardware(comb_hard, is_combination=True)

                # Calculate batchsize under different constraint-bound
                max_batchsize_capacity_bound, max_batchsize_FPRT_bound, max_batchsize_latency_bound = \
                roofline.Get_Max_Batch(hardware=comb_hard["Name"],model=self.model,ratiop=self.ratio_p,maxtoken=self.max_token,\
                    sparse="default",pipeline=1,max_FPRT=3,min_token_per_sec=20,first_col=True,output=True)

                # In serving system, the latency-bound batchsize should not larger than cap-bound results
                if max_batchsize_latency_bound > max_batchsize_capacity_bound:
                    max_batchsize_latency_bound = max_batchsize_capacity_bound

                FPRT, _ = self.cal_metric(ratiop=self.ratio_p, batchsize=max_batchsize_FPRT_bound, maxtoken=self.max_token,\
                    model=self.model, hardware=comb_hard["Name"])
                time_PRF = FPRT

                _, Latency = self.cal_metric(ratiop=self.ratio_p, batchsize=max_batchsize_latency_bound, maxtoken=self.max_token,\
                    model=self.model, hardware=comb_hard["Name"])
                time_GNR = Latency * (self.max_token * (1 - self.ratio_p))

                # The batchsize is Mem-Bound or Latency-Bound
                if max_batchsize_latency_bound == max_batchsize_capacity_bound:
                    status_bs = "Mem-Bound"
                elif max_batchsize_latency_bound < max_batchsize_capacity_bound:
                    status_bs = "Lat-Bound"

                # Calculate how many device needed in PRF phase
                m = int(np.ceil(max_batchsize_latency_bound / max_batchsize_FPRT_bound))

                # Calculate how many device needed in GNR phase
                n = int(np.ceil(time_GNR / time_PRF))

                # Calculate total throught (PRF+GNR) under "m x PRF + n x GNR" system 
                perf = max_batchsize_latency_bound * self.max_token / time_PRF 
                # perf per device; will be discarded
                perf_per_cost = perf / (m + n)

                # Calculte system TCO cost and normalized cost perf token
                roofline.cost_model.init_hardware(roofline.hardwares[comb_hard["Name"]])
                # [TODO] Harecode tradeoff, data from 芯盟
                cap_per_layer = -0.36 * bw/32 + 24.028
                num_of_mem_layer = int(np.ceil( cap / cap_per_layer ))
                
                if self.required_token_per_second == 0:
                    roofline.cost_model.required_token_per_second = 99000000
                else:
                    roofline.cost_model.required_token_per_second = self.required_token_per_second
                price_per_token = roofline.cost_model.cal_system_cost(perf, m, n, num_of_mem_layer, "PRF+GNR-PRJ+GNR-ACT")

                num_THRIVE_chip = roofline.cost_model.XPU_device * roofline.cost_model.XPU.num

                print("\033[1;31m******* Serving System XPU+PIM Config *******\033[0m")
                print("\033[1;31m%-8s%-8s%-8s%-2s%-8s%-8s%-8s%-2s%-8s%-10s%-2s%-12s%-12s%-6s\033[0m"%(
                    "PRF-num", "PRF-bs", "PRF-time", "|",
                    "GNR-num", "GNR-bs", "GNR-time", "|",
                    "Max-bs", "Status-bs", "|", 
                    "Sys-Perf", "Cost/Token", "Impr"))
                print("{:<8}{:<8}{:<8.2f}{:<2}{:<8}{:<8}{:<8.2f}{:<2}{:<8}{:<10}{:<2}{:<12.0f}{:<12.3e}{:<6.2f}".format(
                    m, max_batchsize_FPRT_bound, time_PRF, "|",
                    n, max_batchsize_latency_bound, time_GNR, "|",
                    max_batchsize_capacity_bound, status_bs, "|",
                    perf, price_per_token, self.baseline_price_per_token / price_per_token
                ))

                tflops = roofline.hardwares["THRIVE++"].MM_TFLOPS
                print("\033[1;31m%-7s%-5s%-6s%-10s%-2s%-12s%-14s%-13s%-12s%-12s%-2s%-8s\033[0m"%(
                    "TFLOPS", "BW", "Cap", "Mem-Layer", "|",
                    "Req-Token/s", "Num(T++Chip)", "RMB(T++Chip)", "Num(T++Sys)", "RMB(T++Sys)", "|",
                    "Wafer-Slot"))
                print("{:<7}{:<5.1f}{:<6}{:<10}{:<2}{:<12.2e}{:<13}{:<13}{:<12}{:<12}{:<2}{:<8}".format(
                    tflops, bw, cap, int(num_of_mem_layer), "|",
                    roofline.cost_model.required_token_per_second, int(np.ceil(num_THRIVE_chip)), int(roofline.cost_model.XPU.price), int(np.ceil(roofline.cost_model.XPU_device)), int(roofline.cost_model.XPU_device_price), "|",
                    int(np.ceil(num_THRIVE_chip/66)), "|"
                ))

                self.perf.append(perf)
                self.perf_per_cost.append(perf_per_cost)
                self.price_per_token.append(price_per_token)
                self.improve.append(self.baseline_price_per_token / price_per_token)
                self.num_THRIVE_chip.append(num_THRIVE_chip)
                self.num_of_mem_layer.append(num_of_mem_layer)

if __name__ == "__main__":
    # Model
    model = "GPT3.5"
    ratio_p = 0.5
    max_token = 8192
    batchsize = 32

    list_TFLOPS = [512]
    list_BW_TBs = [512]
    list_PIM_Capacity = [2304]

    # hardware
    new_hard = {
        "Name":"A100*8+PIMv1*1", 
        "PRF":{"hard":"A100", "quant":8},  # V100, A100, H100
        "GNR-PRJ":{"hard":"A100", "quant":8}, 
        "GNR-ACT":{"hard":"PIMv1", "quant":1}, 
        "Server":1,
        "Rack_Unit":4,
        "Assign":"PRF+GNR-PRJ, GNR-ACT"
    }

    pimv1dse = PIMv1DSE(comb_hard=new_hard, batchsize=batchsize, model=model, ratio_p=ratio_p, max_token=max_token, \
        XPU_baseline="A100", list_TFLOPS=list_TFLOPS, list_BW_TBs=list_BW_TBs, list_PIM_Capacity=list_PIM_Capacity)

    pimv1dse.required_token_per_second = 551961
    pimv1dse.cal_baseline_serving_system_perf()
    pimv1dse.cal_serving_system_perf() 
    