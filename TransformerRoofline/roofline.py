import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Any
from openpyxl import load_workbook
import math
from scipy.stats import poisson
from .cost import CostEst
import time
import random

comp_1k = 1024
mem_1k = 1024
pcie_latency = 1500e-9  # 150ns
pcie_BW = 64e-3  # PCIe5.0 x16, 64GB/s


class SparsePattern:
    pass


class Hardware:
    MM_TFLOPS = 0
    MM_BW_TBs = 0
    MM_Sweet_Point = 0
    MM_Card_Num = 0
    MM_GP_TFLOPS = 0
    MM_GP_BW_TBs = 0
    MM_GP_Sweet_Point = 0
    MM_GP_Card_Num = 0
    MV_TFLOPS = 0
    MV_BW_TBs = 0
    MV_Sweet_Point = 0
    MV_Card_Num = 0
    Name = ""
    Form_Type = ""
    Capacity = 0
    MM_Capacity = 0
    MM_GP_Capacity = 0
    MV_Capacity = 0
    Offcard_Type = []
    Assign = ""
    AllReduce_Links = ["NVLink"]
    DDR_Channel = 0
    Pcie = None
    Nvlink = None

    def __init__(self, input_hardware, is_combination=False):
        # is_combination = False, input_hardware is dict
        # is_combination = True, input_hardware = [(Combination_name, (PRF_hardware_name, Quant), (GNR_PRJ_hardware_name, Quant), (GNR_ACT_hardware_name, Quant)), ... ]
        # print(input_hardware)
        if not is_combination:
            self.Name = input_hardware["Name"]
            self.Form_Type = input_hardware["Type"]
            if self.Form_Type == "Homo":
                self.MM_TFLOPS = input_hardware["TFLOPS"]
                self.MM_BW_TBs = input_hardware["BW_TBs"]
                self.MM_Card_Num = input_hardware["Card_Num"]
                self.MM_GP_TFLOPS = input_hardware["TFLOPS"]
                self.MM_GP_BW_TBs = input_hardware["BW_TBs"]
                self.MM_GP_Card_Num = input_hardware["Card_Num"]
                self.MV_TFLOPS = input_hardware["TFLOPS"]
                self.MV_BW_TBs = input_hardware["BW_TBs"]
                self.MV_Card_Num = input_hardware["Card_Num"]
                ##### merge cost parameters
                # self.Name = input_hardware["Name"]
                # self.Form_Type = input_hardware["Type"]
                self.Static_Power = input_hardware["Static_Power"]
                self.TDP = input_hardware["TDP"]
                self.num = self.MV_Card_Num
                self.price = (
                    input_hardware["Price"] if "Price" in input_hardware else None
                )
                self.util = 1.0
                self.TFLOPS = (
                    input_hardware["TFLOPS"] if "TFLOPS" in input_hardware else None
                )
                self.BW_TBs = (
                    input_hardware["BW_TBs"] if "BW_TBs" in input_hardware else None
                )
                self.Capacity = (
                    input_hardware["Capacity"] if "Capacity" in input_hardware else None
                )
                self.Memory_Tech_Node = (
                    input_hardware["Memory_Tech_Node"]
                    if "Memory_Tech_Node" in input_hardware
                    else None
                )
                self.Logic_Tech_Node = (
                    input_hardware["Logic_Tech_Node"]
                    if "Logic_Tech_Node" in input_hardware
                    else None
                )
                self.Memory_Layer = (
                    input_hardware["Memory_Layer"]
                    if "Memory_Layer" in input_hardware
                    else None
                )
                self.Die_Area = (
                    input_hardware["Die_Area"] if "Die_Area" in input_hardware else None
                )
                self.Pcie = input_hardware["pcie"] if "pcie" in input_hardware else None
                self.Nvlink = (
                    input_hardware["nvlink"] if "nvlink" in input_hardware else None
                )
            else:
                self.MM_TFLOPS = input_hardware["MM_TFLOPS"]
                self.MM_BW_TBs = input_hardware["MM_BW_TBs"]
                self.MM_Card_Num = input_hardware["MM_Card_Num"]
                self.MM_GP_TFLOPS = input_hardware["MM_GP_TFLOPS"]
                self.MM_GP_BW_TBs = input_hardware["MM_GP_BW_TBs"]
                self.MM_GP_Card_Num = input_hardware["MM_GP_Card_Num"]
                self.MV_TFLOPS = input_hardware["MV_TFLOPS"]
                self.MV_BW_TBs = input_hardware["MV_BW_TBs"]
                self.MV_Card_Num = input_hardware["MV_Card_Num"]
            self.MM_Sweet_Point = self.MM_TFLOPS / self.MM_BW_TBs
            self.MM_GP_Sweet_Point = self.MM_GP_TFLOPS / self.MM_GP_BW_TBs
            self.MV_Sweet_Point = self.MV_TFLOPS / self.MV_BW_TBs
            if "Capacity" in input_hardware:
                self.Capacity = input_hardware["Capacity"]
            if "MM_Capcity" in input_hardware:
                self.MM_Capacity = input_hardware["MM_Capacity"]
            if "MM_GP_Capacity" in input_hardware:
                self.MM_GP_Capacity = input_hardware["MM_GP_Capacity"]
            if "MV_Capacity" in input_hardware:
                self.MV_Capacity = input_hardware["MV_Capacity"]
                pass
            if "Offcard_Type" in input_hardware:
                self.Offcard_Type = input_hardware["Offcard_Type"]
            if "Assign" in input_hardware:
                self.Assign = input_hardware["Assign"]
            self.AllReduce_Links = (
                input_hardware["AllReduce_Links"]
                if "AllReduce_Links" in input_hardware
                else ["NVLink"]
            )
            self.DDR_Channel = (
                input_hardware["DDR_Channel"]
                if "DDR_Channel" in input_hardware
                else None
            )
        else:
            self.Name = input_hardware["Name"]
            self.Form_Type = (
                "Hete"
                if "Type" not in input_hardware.keys()
                else input_hardware["Type"]
            )
            self.MM_TFLOPS = (
                input_hardware["PRF"]["hard"].MM_TFLOPS * input_hardware["PRF"]["quant"]
            )
            self.MM_BW_TBs = (
                input_hardware["PRF"]["hard"].MM_BW_TBs * input_hardware["PRF"]["quant"]
            )
            self.MM_Capacity = (
                input_hardware["PRF"]["hard"].Capacity * input_hardware["PRF"]["quant"]
            )
            self.MM_Card_Num = input_hardware["PRF"]["quant"]
            self.MM_GP_TFLOPS = (
                input_hardware["GNR-PRJ"]["hard"].MM_GP_TFLOPS
                * input_hardware["GNR-PRJ"]["quant"]
            )
            self.MM_GP_BW_TBs = (
                input_hardware["GNR-PRJ"]["hard"].MM_GP_BW_TBs
                * input_hardware["GNR-PRJ"]["quant"]
            )
            self.MM_GP_Capacity = (
                input_hardware["GNR-PRJ"]["hard"].Capacity
                * input_hardware["GNR-PRJ"]["quant"]
            )
            self.MM_GP_Card_Num = input_hardware["GNR-PRJ"]["quant"]
            self.MV_TFLOPS = (
                input_hardware["GNR-ACT"]["hard"].MV_TFLOPS
                * input_hardware["GNR-ACT"]["quant"]
            )
            self.MV_BW_TBs = (
                input_hardware["GNR-ACT"]["hard"].MV_BW_TBs
                * input_hardware["GNR-ACT"]["quant"]
            )
            self.MV_Capacity = (
                input_hardware["GNR-ACT"]["hard"].Capacity
                * input_hardware["GNR-ACT"]["quant"]
            )
            self.MV_Card_Num = input_hardware["GNR-ACT"]["quant"]
            if "Offcard_Type" in input_hardware:
                self.Offcard_Type = input_hardware["Offcard_Type"]
            if "Assign" in input_hardware:
                self.Assign = input_hardware["Assign"]
            self.MM_Sweet_Point = self.MM_TFLOPS / self.MM_BW_TBs
            self.MM_GP_Sweet_Point = self.MM_GP_TFLOPS / self.MM_GP_BW_TBs
            self.MV_Sweet_Point = self.MV_TFLOPS / self.MV_BW_TBs
            ### merge for cost
            # [FIXME] if using cost, must use combine HW input
            self.XPU = input_hardware["PRF"]["hard"]
            if self.Assign == "PRF+GNR-PRJ+GNR-ACT":
                self.PIM = None
            elif (
                self.Assign == "PRF+GNR-PRJ, GNR-ACT"
                or self.Assign == "PRF, GNR-PRJ+GNR-ACT"
            ):
                self.PIM = input_hardware["GNR-ACT"]["hard"]
            # [FIXME] hard code pim_layer; the pim_layer is related to capacity and TFLOPS
            self.pim_layer = int(input_hardware["GNR-ACT"]["hard"].Capacity / 16)
            self.server_num = (
                input_hardware["Server"] if "Server" in input_hardware else 1
            )
            self.rack_unit = (
                input_hardware["Rack_Unit"] if "Rack_Unit" in input_hardware else 0
            )
            self.AllReduce_Links = (
                input_hardware["AllReduce_Links"]
                if "AllReduce_Links" in input_hardware
                else ["NVLink"]
            )
            self.DDR_Channel = (
                input_hardware["DDR_Channel"]
                if "DDR_Channel" in input_hardware
                else None
            )

    def __str__(self):
        return "\
        Name: {}\n\
        MM_TFLOPS: {}\n\
        MM_BW_TBs: {}\n\
        MM_GP_TFLOPS: {}\n\
        MM_GP_BW_TBs: {}\n\
        MV_TFLOPS: {}\n\
        MV_BW_TBs: {}".format(
            self.Name,
            self.MM_TFLOPS,
            self.MM_BW_TBs,
            self.MM_GP_TFLOPS,
            self.MM_GP_BW_TBs,
            self.MV_TFLOPS,
            self.MV_BW_TBs,
        )


class Model:
    Nhead = 0
    Dmodel = 0
    Dim_Per_Head = 0
    Nlayer = 0
    Name = ""
    Max_Token = 0
    Multi_Query = False
    FFN_Hidden = 0
    FFN_MOE = False
    MOE_Quantity = 1
    MOE_Activate = 1
    Grouped_Query = False
    Grouped_Num = 1

    def __init__(self, input_model):
        self.Name = input_model["Name"]
        self.Nhead = input_model["Nhead"]
        self.Dmodel = input_model["Dmodel"]
        self.Nlayer = input_model["Nlayer"]
        self.Max_Token = input_model["Max_Token"]
        self.Dim_Per_Head = self.Dmodel / self.Nhead
        self.Wnum_in_MLP = input_model["Wnum_in_MLP"]
        self.Multi_Query = (
            True
            if "Multi_Query" in input_model and input_model["Multi_Query"] == "True"
            else False
        )
        self.Grouped_Query = (
            True
            if "Grouped_Query" in input_model and input_model["Grouped_Query"] == "True"
            else False
        )
        self.Parallel_FFN = (
            True
            if "Parallel_FFN" in input_model and input_model["Parallel_FFN"] == "True"
            else False
        )
        self.FFN_Hidden = (
            input_model["FFN_Hidden"]
            if "FFN_Hidden" in input_model
            else 4 * self.Dmodel
        )
        self.FFN_MOE = (
            True
            if "FFN_MOE" in input_model and input_model["FFN_MOE"] == "True"
            else False
        )
        self.MOE_Quantity = (
            input_model["MOE_Quantity"] if "MOE_Quantity" in input_model else 1
        )
        self.MOE_Activate = (
            input_model["MOE_Activate"] if "MOE_Activate" in input_model else 1
        )
        if self.Multi_Query == True:
            self.Grouped_Num = 1
        else:
            self.Grouped_Num = (
                input_model["Grouped_Num"]
                if "Grouped_Num" in input_model
                else self.Nhead
            )

    def Calculate_Arithmetic(
        self,
        Ratio_P,
        Batchsize,
        Max_Token,
        SparsePattern=None,
        Pipeline_Stage=1,
        Assign="PRF+GNR-PRJ, GNR-ACT",
    ):
        # PRF = Prefill, GNR = Generation, PRJ = Projection, ACT = Act-to-Act
        Compute = {
            "PRF": {
                "PRJ": 0,
                "ACT": 0,
                "Shared_PRJ": 0,
                "FFN_PRJ": 0,
            },  # Only useful when self.MOE_Activate != self.MOE_Quantity, have been added in "PRF"
            "GNR": {
                "PRJ": 0,
                "ACT": 0,
                "Shared_PRJ": 0,
                "FFN_PRJ": 0,
            },  # Only useful when self.MOE_Activate != self.MOE_Quantity, have been added in "GNR"
            "Total": 0,
        }
        Memory = {
            "PRF": {
                "PRJ": 0,
                "ACT": 0,
                "Shared_PRJ": 0,
                "FFN_PRJ": 0,
            },  # Only useful when self.MOE_Activate != self.MOE_Quantity, have been added in "PRF"
            "GNR": {
                "PRJ": 0,
                "ACT": 0,
                "Shared_PRJ": 0,
                "FFN_PRJ": 0,
            },  # Only useful when self.MOE_Activate != self.MOE_Quantity, have been added in "GNR"
            "Total": 0,
        }
        Arithmetic = {
            "PRF": 0,
            "GNR": {"PRJ": 0, "ACT": 0},
            "PRF-Shared_PRJ": 0,
            "PRF-FFN_PRJ": 0,
            "GNR-Shared_PRJ": 0,
            "GNR-FFN_PRJ": 0,
            "Total": 0,
        }
        Capacity = {"Weight": 0, "KVCache": 0, "Acti": 0}
        Offcard = {}
        # Offcard_Trans = {"Initial_KV":0, "Acti":0, "Initial_KV_Round":0, "Acti_Round":0}

        Prefill_L = Max_Token * Ratio_P
        Iteration_Num = Max_Token * (1 - Ratio_P)
        Average_KV_L = Max_Token * (1 + Ratio_P) / 2

        Compute["PRF"]["ACT"] += (
            Prefill_L * self.Dim_Per_Head * Prefill_L * self.Nhead * 2 * 2
        )  # Q*KT & Score*V
        Compute["PRF"]["ACT"] *= Batchsize  # Multiply Batchsize
        Compute["PRF"]["ACT"] *= self.Nlayer  # Multiply Layer
        if SparsePattern != None:
            Compute["PRF"]["ACT"] *= SparsePattern.COMP_PRF_ACT  # Sparsity
        Compute["PRF"]["ACT"] /= (
            comp_1k * comp_1k * comp_1k * comp_1k
        )  # Scale Down to TFLOPS

        Compute["PRF"]["PRJ"] += (
            Prefill_L * self.Dmodel * self.Dim_Per_Head * self.Nhead * 2 * 3
        )  # Q/K/V
        Compute["PRF"]["PRJ"] += (
            Prefill_L * self.Dmodel * self.Dmodel * 2
        )  # W0 after attention
        Compute["PRF"]["PRJ"] += (
            Prefill_L * self.Dmodel * self.FFN_Hidden * 2 * 2 * self.MOE_Activate
        )  # MOE FFN
        Compute["PRF"]["PRJ"] *= Batchsize  # Multiply Batchsize
        Compute["PRF"]["PRJ"] *= self.Nlayer  # Multiply Layer
        if SparsePattern != None:
            Compute["PRF"]["PRJ"] *= SparsePattern.COMP_PRF_PRJ  # Sparsity
        Compute["PRF"]["PRJ"] /= (
            comp_1k * comp_1k * comp_1k * comp_1k
        )  # Scale Down to TFLOPS
        # if self.MOE_Activate != self.MOE_Quantity:
        Compute["PRF"]["FFN_PRJ"] += (
            Prefill_L * self.Dmodel * self.FFN_Hidden * 2 * 2 * self.MOE_Activate
        )
        Compute["PRF"]["FFN_PRJ"] *= Batchsize
        Compute["PRF"]["FFN_PRJ"] *= self.Nlayer
        if SparsePattern != None:
            Compute["PRF"]["FFN_PRJ"] *= SparsePattern.COMP_PRF_PRJ
        Compute["PRF"]["FFN_PRJ"] /= comp_1k * comp_1k * comp_1k * comp_1k
        Compute["PRF"]["Shared_PRJ"] = Compute["PRF"]["PRJ"] - Compute["PRF"]["FFN_PRJ"]

        Compute["GNR"]["ACT"] += (
            self.Dim_Per_Head * Average_KV_L * self.Nhead * 2 * 2
        )  # Q*KT & Score*V
        Compute["GNR"]["ACT"] *= Batchsize  # Multiply Batchsize
        Compute["GNR"]["ACT"] *= Iteration_Num  # Multiply Iteration Number
        Compute["GNR"]["ACT"] *= self.Nlayer  # Multiply Layer
        if SparsePattern != None:
            Compute["GNR"]["ACT"] *= SparsePattern.COMP_GNR_ACT  # Sparsity
        Compute["GNR"]["ACT"] /= (
            comp_1k * comp_1k * comp_1k * comp_1k
        )  # Scale Down to TFLOPS

        Compute["GNR"]["PRJ"] += (
            self.Dmodel * self.Dim_Per_Head * self.Nhead * 2 * 3
        )  # Q/K/V
        Compute["GNR"]["PRJ"] += self.Dmodel * self.Dmodel * 2  # W0 after attention
        Compute["GNR"]["PRJ"] += (
            self.Dmodel * self.FFN_Hidden * 2 * 2 * self.MOE_Activate
        )  # FFN
        Compute["GNR"]["PRJ"] *= Batchsize  # Multiply Batchsize
        Compute["GNR"]["PRJ"] *= Iteration_Num  # Multiply Iteration Number
        Compute["GNR"]["PRJ"] *= self.Nlayer  # Multiply Layer
        if SparsePattern != None:
            Compute["GNR"]["PRJ"] *= SparsePattern.COMP_GNR_PRJ  # Sparsity
        Compute["GNR"]["PRJ"] /= (
            comp_1k * comp_1k * comp_1k * comp_1k
        )  # Scale Down to TFLOPS
        # if self.MOE_Activate != self.MOE_Quantity:
        Compute["GNR"]["FFN_PRJ"] += (
            self.Dmodel * self.FFN_Hidden * 2 * 2 * self.MOE_Activate
        )
        Compute["GNR"]["FFN_PRJ"] *= Batchsize
        Compute["GNR"]["FFN_PRJ"] *= Iteration_Num
        Compute["GNR"]["FFN_PRJ"] *= self.Nlayer
        if SparsePattern != None:
            Compute["GNR"]["FFN_PRJ"] *= SparsePattern.COMP_PRF_PRJ
        Compute["GNR"]["FFN_PRJ"] /= comp_1k * comp_1k * comp_1k * comp_1k
        Compute["GNR"]["Shared_PRJ"] = Compute["GNR"]["PRJ"] - Compute["GNR"]["FFN_PRJ"]
        # print(
        #     f'{Compute["GNR"]["Shared_PRJ"]=}, {Compute["GNR"]["PRJ"]=} {Compute["GNR"]["FFN_PRJ"]=}'
        # )

        Compute["Total"] = (
            Compute["PRF"]["PRJ"]
            + Compute["PRF"]["ACT"]
            + Compute["GNR"]["PRJ"]
            + Compute["GNR"]["ACT"]
        )

        Memory["PRF"][
            "ACT"
        ] += 0  # Assume all activations can be stored in on-chip SRAM

        Memory["PRF"]["PRJ"] += (
            self.Dmodel * self.Dim_Per_Head * self.Nhead * 2 * 3
        )  # Weight for Q/K/V
        Memory["PRF"]["PRJ"] += self.Dmodel * self.Dmodel * 2  # W0 after attention
        Memory["PRF"]["PRJ"] += (
            self.Dmodel * self.FFN_Hidden * 2 * 2 * self.MOE_Activate
        )  # MOE FFN Weight
        Memory["PRF"]["PRJ"] *= self.Nlayer  # Multiply Layer
        if SparsePattern != None:
            Memory["PRF"]["PRJ"] *= SparsePattern.MEM_PRF_PRJ  # Sparsity
        Memory["PRF"]["PRJ"] /= mem_1k * mem_1k * mem_1k * mem_1k  # Scale Down to TB
        # if self.MOE_Activate != self.MOE_Quantity:
        Memory["PRF"]["FFN_PRJ"] += (
            self.Dmodel * self.FFN_Hidden * 2 * 2 * self.MOE_Activate
        )
        Memory["PRF"]["FFN_PRJ"] *= self.Nlayer
        if SparsePattern != None:
            Memory["PRF"]["FFN_PRJ"] *= SparsePattern.COMP_PRF_PRJ
        Memory["PRF"]["FFN_PRJ"] /= mem_1k * mem_1k * mem_1k * mem_1k
        Memory["PRF"]["Shared_PRJ"] = Memory["PRF"]["PRJ"] - Memory["PRF"]["FFN_PRJ"]

        Memory["GNR"]["ACT"] += (
            self.Dim_Per_Head * Average_KV_L * self.Nhead * 2 * 2
        )  # KV Cache
        Memory["GNR"]["ACT"] *= Batchsize  # Multiply Batchsize
        Memory["GNR"]["ACT"] *= Iteration_Num  # Multiply Iteration Number
        if SparsePattern != None:
            Memory["GNR"]["ACT"] *= SparsePattern.MEM_GNR_ACT  # Sparsity
        # if self.Multi_Query: Memory["GNR"]["ACT"] /= self.Nhead
        Memory["GNR"]["ACT"] /= (
            self.Nhead / self.Grouped_Num
        )  # Multi-Query or Multi-Group
        Memory["GNR"]["ACT"] *= self.Nlayer  # Multiply Layer
        Memory["GNR"]["ACT"] /= mem_1k * mem_1k * mem_1k * mem_1k  # Scale Down to TB

        Memory["GNR"]["PRJ"] += (
            self.Dmodel * self.Dim_Per_Head * self.Nhead * 2 * 3
        )  # Weight for Q/K/V
        Memory["GNR"]["PRJ"] += self.Dmodel * self.Dmodel * 2  # W0 after attention
        Memory["GNR"]["PRJ"] += (
            self.Dmodel * self.FFN_Hidden * 2 * 2 * self.MOE_Activate
        )  # FFN Weight
        Memory["GNR"]["PRJ"] *= Iteration_Num  # Multiply Iteration Number
        Memory["GNR"]["PRJ"] *= self.Nlayer  # Multiply Layer
        if SparsePattern != None:
            Memory["GNR"]["PRJ"] *= SparsePattern.MEM_GNR_PRJ  # Sparsity
        Memory["GNR"]["PRJ"] /= mem_1k * mem_1k * mem_1k * mem_1k  # Scale Down to TB
        # if self.MOE_Activate != self.MOE_Quantity:
        Memory["GNR"]["FFN_PRJ"] += (
            self.Dmodel * self.FFN_Hidden * 2 * 2 * self.MOE_Activate
        )
        Memory["GNR"]["FFN_PRJ"] *= Iteration_Num
        Memory["GNR"]["FFN_PRJ"] *= self.Nlayer
        if SparsePattern != None:
            Memory["GNR"]["FFN_PRJ"] *= SparsePattern.COMP_PRF_PRJ
        Memory["GNR"]["FFN_PRJ"] /= mem_1k * mem_1k * mem_1k * mem_1k
        Memory["GNR"]["Shared_PRJ"] = Memory["GNR"]["PRJ"] - Memory["GNR"]["FFN_PRJ"]

        Memory["Total"] = (
            Memory["PRF"]["PRJ"]
            + Memory["PRF"]["ACT"]
            + Memory["GNR"]["PRJ"]
            + Memory["GNR"]["ACT"]
        )

        Arithmetic["PRF"] = (Compute["PRF"]["PRJ"] + Compute["PRF"]["ACT"]) / (
            Memory["PRF"]["PRJ"] + Memory["PRF"]["ACT"]
        )
        Arithmetic["PRF-PRJ"] = Compute["PRF"]["PRJ"] / Memory["PRF"]["PRJ"]
        Arithmetic["PRF-ACT"] = 1e6  # No DRAM transmission, always compute bound
        Arithmetic["GNR"]["PRJ"] = (
            0
            if Compute["GNR"]["PRJ"] == 0
            else Compute["GNR"]["PRJ"] / Memory["GNR"]["PRJ"]
        )
        Arithmetic["GNR"]["ACT"] = (
            0
            if Compute["GNR"]["ACT"] == 0
            else Compute["GNR"]["ACT"] / Memory["GNR"]["ACT"]
        )
        Arithmetic["PRF-Shared_PRJ"] = (
            Compute["PRF"]["Shared_PRJ"] / Memory["PRF"]["Shared_PRJ"]
        )
        Arithmetic["PRF-FFN_PRJ"] = Compute["PRF"]["FFN_PRJ"] / Memory["PRF"]["FFN_PRJ"]
        Arithmetic["GNR-Shared_PRJ"] = (
            Compute["GNR"]["Shared_PRJ"] / Memory["GNR"]["Shared_PRJ"]
        )
        Arithmetic["GNR-FFN_PRJ"] = Compute["GNR"]["FFN_PRJ"] / Memory["GNR"]["FFN_PRJ"]
        Arithmetic["Total"] = Compute["Total"] / Memory["Total"]

        Capacity["Weight"] = Memory["PRF"]["PRJ"] * mem_1k  # GB
        Capacity["KVCache"] = (
            0
            if Memory["GNR"]["ACT"] == 0
            else Memory["GNR"]["ACT"] / ((1 + Ratio_P) / 2) / Iteration_Num * mem_1k
        )  # GB
        Capacity["Acti"] = (
            self.Dmodel * Prefill_L * Batchsize * 2 / (mem_1k * mem_1k * mem_1k)
        )  # GB

        Offcard_tmp = {
            "PRF": {
                "inter_Stage": [0, 0],
                "intra_Stage": [0, 0],
                "out_Card": [0, 0],
            },  # [Size per transfer, transfer amount]
            "GNR-PRJ": {
                "inter_Stage": [0, 0],
                "intra_Stage": [0, 0],
                "out_Card": [0, 0],
            },
            "GNR-ACT": {
                "inter_Stage": [0, 0],
                "intra_Stage": [0, 0],
                "out_Card": [0, 0],
            },
        }
        # Pipeline_Stage = Parallel[1]
        Offcard_tmp["PRF"]["inter_Stage"][0] += Prefill_L * self.Dmodel * 2
        Offcard_tmp["PRF"]["inter_Stage"][0] *= Batchsize  # Multiply Batchszie
        ### Activation Sparsity?
        if Pipeline_Stage > 1:
            Offcard_tmp["PRF"]["inter_Stage"][
                1
            ] += Pipeline_Stage  # or Pipeline_Stage - 1?
        else:
            Offcard_tmp["PRF"]["inter_Stage"][
                1
            ] = 0  # If only one pipeline stage, no inter_stage communication

        Offcard_tmp["PRF"]["intra_Stage"][0] += Prefill_L * self.Dmodel * 2
        Offcard_tmp["PRF"]["intra_Stage"][0] *= Batchsize  # Multiply Batchsize
        ### Activation Sparsity?
        Offcard_tmp["PRF"]["intra_Stage"][1] += (
            2 * self.Nlayer
        )  # two transfer per layer

        Offcard_tmp["PRF"]["out_Card"][0] += 0
        Offcard_tmp["PRF"]["out_Card"][1] += 0

        Offcard_tmp["GNR-PRJ"]["inter_Stage"][0] += self.Dmodel * 2
        Offcard_tmp["GNR-PRJ"]["inter_Stage"][0] *= Batchsize  # Multiply Batchsize
        ### Activation Sparsity?
        if Pipeline_Stage > 1:
            Offcard_tmp["GNR-PRJ"]["inter_Stage"][1] += Pipeline_Stage
        else:
            Offcard_tmp["GNR-PRJ"]["inter_Stage"][
                1
            ] += 0  # If only one pipeline stage, no inter_stage communication
        Offcard_tmp["GNR-PRJ"]["inter_Stage"][1] *= Iteration_Num

        Offcard_tmp["GNR-PRJ"]["intra_Stage"][0] += self.Dmodel * 2
        Offcard_tmp["GNR-PRJ"]["intra_Stage"][0] *= Batchsize  # Multiply Batchsize
        ### Activation Sparsity?
        Offcard_tmp["GNR-PRJ"]["intra_Stage"][1] += (
            2 * self.Nlayer
        )  # Two transfer per lyaer
        Offcard_tmp["GNR-PRJ"]["intra_Stage"][1] *= Iteration_Num

        # print(Offcard_tmp["GNR-PRJ"]["intra_Stage"])

        Offcard_tmp["GNR-ACT"]["inter_Stage"][0] += 0
        Offcard_tmp["GNR-ACT"]["inter_Stage"][1] += 0

        Offcard_tmp["GNR-ACT"]["intra_Stage"][0] += 0
        Offcard_tmp["GNR-ACT"]["intra_Stage"][1] += 0

        Offcard_tmp["GNR-PRJ"]["out_Card"][0] += 0
        Offcard_tmp["GNR-PRJ"]["out_Card"][1] += 0

        if Assign == "PRF+GNR-PRJ+GNR-ACT":
            Offcard_tmp["GNR-ACT"]["out_Card"][0] += 0
            Offcard_tmp["GNR-ACT"]["out_Card"][1] += 0
        elif Assign == "PRF+GNR-PRJ, GNR-ACT":
            Offcard_tmp["GNR-ACT"]["out_Card"][0] += self.Dmodel * 2
            Offcard_tmp["GNR-ACT"]["out_Card"][0] *= Batchsize
            ### Activation Sparsity?
            Offcard_tmp["GNR-ACT"]["out_Card"][1] += 2 * self.Nlayer
            Offcard_tmp["GNR-ACT"]["out_Card"][1] *= Iteration_Num

        Offcard = Offcard_tmp

        """
        Offcard_Trans["Initial_KV"] += self.Dmodel * Prefill_L * 2 * 2 # KV Cache after Prefill
        Offcard_Trans["Initial_KV"] *= Batchsize # Multiply Batchsize
        Offcard_Trans["Initial_KV"] /= (mem_1k * mem_1k * mem_1k * mem_1k) # Scale down to TB
        Offcard_Trans["Initial_KV_Round"] = 0

        Offcard_Trans["Acti"] += self.Dmodel * 2 * 3 # Q/K/V
        Offcard_Trans["Acti"] *= 2 # Two transfer, GPU -> PIM and PIM -> GPU
        Offcard_Trans["Acti"] *= Batchsize # Multiply Batchsize
        Offcard_Trans["Acti"] *= self.Nlayer # Multiply Layer
        Offcard_Trans["Acti"] *= Iteration_Num # Multiply Iteration Number
        Offcard_Trans["Acti"] /= (mem_1k * mem_1k * mem_1k * mem_1k) # Scale down to TB
        Offcard_Trans["Acti_Round"] = self.Nlayer * 2 * Iteration_Num
        """
        return Compute, Memory, Arithmetic, Capacity, Offcard  # Offcard_Trans

    def Calculate_Arithmetic_Iteration(
        self,
        Prompt_Len,
        Step,
        Batchsize,
        SparsePattern=None,
        Pipeline_Stage=1,
        Assign="PRF+GNR-PRJ, GNR-ACT",
    ):
        # PRF = Prefill, GNR = Generation, PRJ = Projection, ACT = Act-to-Act
        Compute = {
            "PRF": {
                "PRJ": 0,
                "ACT": 0,
                "Shared_PRJ": 0,
                "FFN_PRJ": 0,
            },  # Only useful when self.MOE_Activate != self.MOE_Quantity, have been added in "PRF"
            "GNR": {
                "PRJ": 0,
                "ACT": 0,
                "Shared_PRJ": 0,
                "FFN_PRJ": 0,
            },  # Only useful when self.MOE_Activate != self.MOE_Quantity, have been added in "GNR"
            "Total": 0,
        }
        Memory = {
            "PRF": {
                "PRJ": 0,
                "ACT": 0,
                "Shared_PRJ": 0,
                "FFN_PRJ": 0,
            },  # Only useful when self.MOE_Activate != self.MOE_Quantity, have been added in "PRF"
            "GNR": {
                "PRJ": 0,
                "ACT": 0,
                "Shared_PRJ": 0,
                "FFN_PRJ": 0,
            },  # Only useful when self.MOE_Activate != self.MOE_Quantity, have been added in "GNR"
            "Total": 0,
        }
        Arithmetic = {
            "PRF": 0,
            "GNR": {"PRJ": 0, "ACT": 0},
            "PRF-Shared_PRJ": 0,
            "PRF-FFN_PRJ": 0,
            "GNR-Shared_PRJ": 0,
            "GNR-FFN_PRJ": 0,
            "Total": 0,
        }
        Capacity = {"Weight": 0, "KVCache": 0, "Acti": 0}
        Offcard = {}
        # Offcard_Trans = {"Initial_KV":0, "Acti":0, "Initial_KV_Round":0, "Acti_Round":0}

        Prefill_L = Prompt_Len + Step
        Iteration_Num = 1
        Average_KV_L = Prompt_Len + Step

        if Step == 0:
            Compute["PRF"]["ACT"] += (
                Prefill_L * self.Dim_Per_Head * Prefill_L * self.Nhead * 2 * 2
            )  # Q*KT & Score*V
            Compute["PRF"]["ACT"] *= Batchsize  # Multiply Batchsize
            Compute["PRF"]["ACT"] *= self.Nlayer  # Multiply Layer
            if SparsePattern != None:
                Compute["PRF"]["ACT"] *= SparsePattern.COMP_PRF_ACT  # Sparsity
            Compute["PRF"]["ACT"] /= (
                comp_1k * comp_1k * comp_1k * comp_1k
            )  # Scale Down to TFLOPS

            Compute["PRF"]["PRJ"] += (
                Prefill_L * self.Dmodel * self.Dim_Per_Head * self.Nhead * 2 * 3
            )  # Q/K/V
            Compute["PRF"]["PRJ"] += (
                Prefill_L * self.Dmodel * self.Dmodel * 2
            )  # W0 after attention
            Compute["PRF"]["PRJ"] += (
                Prefill_L
                * self.Dmodel
                * self.FFN_Hidden
                * self.Wnum_in_MLP
                * 2
                * self.MOE_Activate
            )  # MOE FFN
            Compute["PRF"]["PRJ"] *= Batchsize  # Multiply Batchsize
            Compute["PRF"]["PRJ"] *= self.Nlayer  # Multiply Layer
            if SparsePattern != None:
                Compute["PRF"]["PRJ"] *= SparsePattern.COMP_PRF_PRJ  # Sparsity
            Compute["PRF"]["PRJ"] /= (
                comp_1k * comp_1k * comp_1k * comp_1k
            )  # Scale Down to TFLOPS
            # if self.MOE_Activate != self.MOE_Quantity:
            Compute["PRF"]["FFN_PRJ"] += (
                Prefill_L * self.Dmodel * self.FFN_Hidden * 2 * 2 * self.MOE_Activate
            )
            Compute["PRF"]["FFN_PRJ"] *= Batchsize
            Compute["PRF"]["FFN_PRJ"] *= self.Nlayer
            if SparsePattern != None:
                Compute["PRF"]["FFN_PRJ"] *= SparsePattern.COMP_PRF_PRJ
            Compute["PRF"]["FFN_PRJ"] /= comp_1k * comp_1k * comp_1k * comp_1k
            Compute["PRF"]["Shared_PRJ"] = (
                Compute["PRF"]["PRJ"] - Compute["PRF"]["FFN_PRJ"]
            )

        if Step > 0:
            Compute["GNR"]["ACT"] += (
                self.Dim_Per_Head * Average_KV_L * self.Nhead * 2 * 2
            )  # Q*KT & Score*V
            # print(
            #     f"{self.Dim_Per_Head=}, {Average_KV_L=}, {self.Nhead=}, {Compute['GNR']['ACT']=}"
            # )
            Compute["GNR"]["ACT"] *= Batchsize  # Multiply Batchsize
            # print(f"{Batchsize=}, {Compute['GNR']['ACT']=}")
            Compute["GNR"]["ACT"] *= Iteration_Num  # Multiply Iteration Number
            # print(f"{Iteration_Num=}, {Compute['GNR']['ACT']=}")
            Compute["GNR"]["ACT"] *= self.Nlayer  # Multiply Layer
            # print(f"{self.Nlayer=}, {Compute['GNR']['ACT']=}")
            if SparsePattern != None:
                Compute["GNR"]["ACT"] *= SparsePattern.COMP_GNR_ACT  # Sparsity
            Compute["GNR"]["ACT"] /= (
                comp_1k * comp_1k * comp_1k * comp_1k
            )  # Scale Down to TFLOPS
            # print(f"{Compute['GNR']['ACT']=}")

            Compute["GNR"]["PRJ"] += (
                self.Dmodel * self.Dim_Per_Head * self.Nhead * 2 * 3
            )  # Q/K/V
            # print(
            #     f"{self.Dmodel=}, {self.Dim_Per_Head=}, {self.Nhead=}, {Compute['GNR']['PRJ']=}"
            # )
            Compute["GNR"]["PRJ"] += self.Dmodel * self.Dmodel * 2  # W0 after attention
            # print(f"{self.Dmodel=}, {self.Dmodel=}, {Compute['GNR']['PRJ']=}")
            Compute["GNR"]["PRJ"] += (
                self.Dmodel * self.FFN_Hidden * self.Wnum_in_MLP * 2 * self.MOE_Activate
            )  # FFN
            # print(
            #     f"{self.Dmodel=}, {self.FFN_Hidden=}, {self.MOE_Activate=}, {Compute['GNR']['PRJ']=}"
            # )
            Compute["GNR"]["PRJ"] *= Batchsize  # Multiply Batchsize
            # print(f"{Batchsize=}, {Compute['GNR']['PRJ']=}")
            Compute["GNR"]["PRJ"] *= Iteration_Num  # Multiply Iteration Number
            # print(f"{Iteration_Num=}, {Compute['GNR']['PRJ']=}")
            Compute["GNR"]["PRJ"] *= self.Nlayer  # Multiply Layer
            # print(f"{self.Nlayer=}, {Compute['GNR']['PRJ']=}")
            if SparsePattern != None:
                Compute["GNR"]["PRJ"] *= SparsePattern.COMP_GNR_PRJ  # Sparsity
            Compute["GNR"]["PRJ"] /= (
                comp_1k * comp_1k * comp_1k * comp_1k
            )  # Scale Down to TFLOPS
            # print(f"{Compute['GNR']['PRJ']=}")
            # if self.MOE_Activate != self.MOE_Quantity:
            Compute["GNR"]["FFN_PRJ"] += (
                self.Dmodel * self.FFN_Hidden * 2 * 2 * self.MOE_Activate
            )
            # print(
            #     f"{self.Dmodel=}, {self.FFN_Hidden=}, {self.MOE_Activate=}, {Compute['GNR']['FFN_PRJ']=}"
            # )
            Compute["GNR"]["FFN_PRJ"] *= Batchsize
            # print(f"{Batchsize=}, {Compute['GNR']['FFN_PRJ']=}")
            Compute["GNR"]["FFN_PRJ"] *= Iteration_Num
            # print(f"{Iteration_Num=}, {Compute['GNR']['FFN_PRJ']=}")
            Compute["GNR"]["FFN_PRJ"] *= self.Nlayer
            # print(f"{self.Nlayer=}, {Compute['GNR']['FFN_PRJ']=}")
            if SparsePattern != None:
                Compute["GNR"]["FFN_PRJ"] *= SparsePattern.COMP_PRF_PRJ
            Compute["GNR"]["FFN_PRJ"] /= comp_1k * comp_1k * comp_1k * comp_1k
            # print(f"{Compute['GNR']['FFN_PRJ']=}")
            Compute["GNR"]["Shared_PRJ"] = (
                Compute["GNR"]["PRJ"] - Compute["GNR"]["FFN_PRJ"]
            )

        Compute["Total"] = (
            Compute["PRF"]["PRJ"]
            + Compute["PRF"]["ACT"]
            + Compute["GNR"]["PRJ"]
            + Compute["GNR"]["ACT"]
        )

        if Step == 0:
            Memory["PRF"][
                "ACT"
            ] += 0  # Assume all activations can be stored in on-chip SRAM

            Memory["PRF"]["PRJ"] += (
                self.Dmodel * self.Dim_Per_Head * self.Nhead * 2 * 3
            )  # Weight for Q/K/V
            Memory["PRF"]["PRJ"] += self.Dmodel * self.Dmodel * 2  # W0 after attention
            Memory["PRF"]["PRJ"] += (
                self.Dmodel * self.FFN_Hidden * self.Wnum_in_MLP * 2 * self.MOE_Activate
            )  # MOE FFN Weight
            Memory["PRF"]["PRJ"] *= self.Nlayer  # Multiply Layer
            if SparsePattern != None:
                Memory["PRF"]["PRJ"] *= SparsePattern.MEM_PRF_PRJ  # Sparsity
            Memory["PRF"]["PRJ"] /= (
                mem_1k * mem_1k * mem_1k * mem_1k
            )  # Scale Down to TB
            # if self.MOE_Activate != self.MOE_Quantity:
            Memory["PRF"]["FFN_PRJ"] += (
                self.Dmodel * self.FFN_Hidden * 2 * 2 * self.MOE_Activate
            )
            Memory["PRF"]["FFN_PRJ"] *= self.Nlayer
            if SparsePattern != None:
                Memory["PRF"]["FFN_PRJ"] *= SparsePattern.COMP_PRF_PRJ
            Memory["PRF"]["FFN_PRJ"] /= mem_1k * mem_1k * mem_1k * mem_1k
            Memory["PRF"]["Shared_PRJ"] = (
                Memory["PRF"]["PRJ"] - Memory["PRF"]["FFN_PRJ"]
            )

        if Step > 0:
            Memory["GNR"]["ACT"] += (
                self.Dim_Per_Head * Average_KV_L * self.Nhead * 2 * 2
            )  # KV Cache
            Memory["GNR"]["ACT"] *= Batchsize  # Multiply Batchsize
            Memory["GNR"]["ACT"] *= Iteration_Num  # Multiply Iteration Number
            if SparsePattern != None:
                Memory["GNR"]["ACT"] *= SparsePattern.MEM_GNR_ACT  # Sparsity
            # if self.Multi_Query: Memory["GNR"]["ACT"] /= self.Nhead
            Memory["GNR"]["ACT"] /= (
                self.Nhead / self.Grouped_Num
            )  # Multi-Query or Multi-Group
            # print(f"{self.Nhead=}, {self.Grouped_Num=}, {Memory['GNR']['ACT']=}")
            Memory["GNR"]["ACT"] *= self.Nlayer  # Multiply Layer
            Memory["GNR"]["ACT"] /= (
                mem_1k * mem_1k * mem_1k * mem_1k
            )  # Scale Down to TB

            Memory["GNR"]["PRJ"] += (
                self.Dmodel * self.Dim_Per_Head * self.Nhead * 2 * 3
            )  # Weight for Q/K/V
            Memory["GNR"]["PRJ"] += self.Dmodel * self.Dmodel * 2  # W0 after attention
            Memory["GNR"]["PRJ"] += (
                self.Dmodel * self.FFN_Hidden * self.Wnum_in_MLP * 2 * self.MOE_Activate
            )  # FFN Weight
            Memory["GNR"]["PRJ"] *= Iteration_Num  # Multiply Iteration Number
            Memory["GNR"]["PRJ"] *= self.Nlayer  # Multiply Layer
            # print(f"{Memory['GNR']['PRJ']=}")
            if SparsePattern != None:
                Memory["GNR"]["PRJ"] *= SparsePattern.MEM_GNR_PRJ  # Sparsity
            Memory["GNR"]["PRJ"] /= (
                mem_1k * mem_1k * mem_1k * mem_1k
            )  # Scale Down to TB
            # if self.MOE_Activate != self.MOE_Quantity:
            Memory["GNR"]["FFN_PRJ"] += (
                self.Dmodel * self.FFN_Hidden * 2 * 2 * self.MOE_Activate
            )
            Memory["GNR"]["FFN_PRJ"] *= Iteration_Num
            Memory["GNR"]["FFN_PRJ"] *= self.Nlayer
            if SparsePattern != None:
                Memory["GNR"]["FFN_PRJ"] *= SparsePattern.COMP_PRF_PRJ
            Memory["GNR"]["FFN_PRJ"] /= mem_1k * mem_1k * mem_1k * mem_1k
            Memory["GNR"]["Shared_PRJ"] = (
                Memory["GNR"]["PRJ"] - Memory["GNR"]["FFN_PRJ"]
            )

        Memory["Total"] = (
            Memory["PRF"]["PRJ"]
            + Memory["PRF"]["ACT"]
            + Memory["GNR"]["PRJ"]
            + Memory["GNR"]["ACT"]
        )

        if Step == 0:
            Arithmetic["PRF"] = (Compute["PRF"]["PRJ"] + Compute["PRF"]["ACT"]) / (
                Memory["PRF"]["PRJ"] + Memory["PRF"]["ACT"]
            )
            Arithmetic["PRF-PRJ"] = Compute["PRF"]["PRJ"] / Memory["PRF"]["PRJ"]
            Arithmetic["PRF-ACT"] = 1e6  # No DRAM transmission, always compute bound
            Arithmetic["PRF-Shared_PRJ"] = (
                Compute["PRF"]["Shared_PRJ"] / Memory["PRF"]["Shared_PRJ"]
            )
            Arithmetic["PRF-FFN_PRJ"] = (
                Compute["PRF"]["FFN_PRJ"] / Memory["PRF"]["FFN_PRJ"]
            )
        if Step > 0:
            Arithmetic["GNR"]["PRJ"] = (
                0
                if Compute["GNR"]["PRJ"] == 0
                else Compute["GNR"]["PRJ"] / Memory["GNR"]["PRJ"]
            )
            # print(
            #     f"{Compute['GNR']['PRJ']=}, {Memory['GNR']['PRJ']=} {Arithmetic['GNR']['PRJ']=}"
            # )
            Arithmetic["GNR"]["ACT"] = (
                0
                if Compute["GNR"]["ACT"] == 0
                else Compute["GNR"]["ACT"] / Memory["GNR"]["ACT"]
            )
            # print(
            #     f"{Compute['GNR']['ACT']=}, {Memory['GNR']['ACT']=} {Arithmetic['GNR']['ACT']=}"
            # )
            Arithmetic["GNR-Shared_PRJ"] = (
                Compute["GNR"]["Shared_PRJ"] / Memory["GNR"]["Shared_PRJ"]
            )
            # print(
            #     f"{Compute['GNR']['Shared_PRJ']=}, {Memory['GNR']['Shared_PRJ']=} {Arithmetic['GNR-Shared_PRJ']=}"
            # )
            Arithmetic["GNR-FFN_PRJ"] = (
                Compute["GNR"]["FFN_PRJ"] / Memory["GNR"]["FFN_PRJ"]
            )
            # print(
            #     f"{Compute['GNR']['FFN_PRJ']=}, {Memory['GNR']['FFN_PRJ']=} {Arithmetic['GNR-FFN_PRJ']=}"
            # )
        Arithmetic["Total"] = Compute["Total"] / Memory["Total"]

        Capacity["Weight"] = Memory["PRF"]["PRJ"] * mem_1k  # GB
        Capacity["KVCache"] = (
            0
            if Memory["GNR"]["ACT"] == 0
            else Memory["GNR"]["ACT"] / Iteration_Num * mem_1k
        )  # GB
        Capacity["Acti"] = (
            self.Dmodel * Prefill_L * Batchsize * 2 / (mem_1k * mem_1k * mem_1k)
        )  # GB

        Offcard_tmp = {
            "PRF": {
                "inter_Stage": [0, 0],
                "intra_Stage": [0, 0],
                "out_Card": [0, 0],
            },  # [Size per transfer, transfer amount]
            "GNR-PRJ": {
                "inter_Stage": [0, 0],
                "intra_Stage": [0, 0],
                "out_Card": [0, 0],
            },
            "GNR-ACT": {
                "inter_Stage": [0, 0],
                "intra_Stage": [0, 0],
                "out_Card": [0, 0],
            },
        }
        # Pipeline_Stage = Parallel[1]
        Offcard_tmp["PRF"]["inter_Stage"][0] += Prefill_L * self.Dmodel * 2
        Offcard_tmp["PRF"]["inter_Stage"][0] *= Batchsize  # Multiply Batchszie
        ### Activation Sparsity?
        if Pipeline_Stage > 1:
            Offcard_tmp["PRF"]["inter_Stage"][
                1
            ] += Pipeline_Stage  # or Pipeline_Stage - 1?
        else:
            Offcard_tmp["PRF"]["inter_Stage"][
                1
            ] = 0  # If only one pipeline stage, no inter_stage communication

        Offcard_tmp["PRF"]["intra_Stage"][0] += Prefill_L * self.Dmodel * 2
        Offcard_tmp["PRF"]["intra_Stage"][0] *= Batchsize  # Multiply Batchsize
        ### Activation Sparsity?
        Offcard_tmp["PRF"]["intra_Stage"][1] += (
            2 * self.Nlayer
        )  # two transfer per layer

        Offcard_tmp["PRF"]["out_Card"][0] += 0
        Offcard_tmp["PRF"]["out_Card"][1] += 0

        Offcard_tmp["GNR-PRJ"]["inter_Stage"][0] += self.Dmodel * 2
        Offcard_tmp["GNR-PRJ"]["inter_Stage"][0] *= Batchsize  # Multiply Batchsize
        ### Activation Sparsity?
        if Pipeline_Stage > 1:
            Offcard_tmp["GNR-PRJ"]["inter_Stage"][1] += Pipeline_Stage
        else:
            Offcard_tmp["GNR-PRJ"]["inter_Stage"][
                1
            ] += 0  # If only one pipeline stage, no inter_stage communication
        Offcard_tmp["GNR-PRJ"]["inter_Stage"][1] *= Iteration_Num

        Offcard_tmp["GNR-PRJ"]["intra_Stage"][0] += self.Dmodel * 2
        Offcard_tmp["GNR-PRJ"]["intra_Stage"][0] *= Batchsize  # Multiply Batchsize
        ### Activation Sparsity?
        Offcard_tmp["GNR-PRJ"]["intra_Stage"][1] += (
            2 * self.Nlayer
        )  # Two transfer per lyaer
        Offcard_tmp["GNR-PRJ"]["intra_Stage"][1] *= Iteration_Num

        # print(Offcard_tmp["GNR-PRJ"]["intra_Stage"])

        Offcard_tmp["GNR-ACT"]["inter_Stage"][0] += 0
        Offcard_tmp["GNR-ACT"]["inter_Stage"][1] += 0

        Offcard_tmp["GNR-ACT"]["intra_Stage"][0] += 0
        Offcard_tmp["GNR-ACT"]["intra_Stage"][1] += 0

        Offcard_tmp["GNR-PRJ"]["out_Card"][0] += 0
        Offcard_tmp["GNR-PRJ"]["out_Card"][1] += 0

        if Assign == "PRF+GNR-PRJ+GNR-ACT":
            Offcard_tmp["GNR-ACT"]["out_Card"][0] += 0
            Offcard_tmp["GNR-ACT"]["out_Card"][1] += 0
        elif Assign == "PRF+GNR-PRJ, GNR-ACT":
            Offcard_tmp["GNR-ACT"]["out_Card"][0] += self.Dmodel * 2
            Offcard_tmp["GNR-ACT"]["out_Card"][0] *= Batchsize
            ### Activation Sparsity?
            Offcard_tmp["GNR-ACT"]["out_Card"][1] += 2 * self.Nlayer
            Offcard_tmp["GNR-ACT"]["out_Card"][1] *= Iteration_Num

        Offcard = Offcard_tmp

        """
        Offcard_Trans["Initial_KV"] += self.Dmodel * Prefill_L * 2 * 2 # KV Cache after Prefill
        Offcard_Trans["Initial_KV"] *= Batchsize # Multiply Batchsize
        Offcard_Trans["Initial_KV"] /= (mem_1k * mem_1k * mem_1k * mem_1k) # Scale down to TB
        Offcard_Trans["Initial_KV_Round"] = 0

        Offcard_Trans["Acti"] += self.Dmodel * 2 * 3 # Q/K/V
        Offcard_Trans["Acti"] *= 2 # Two transfer, GPU -> PIM and PIM -> GPU
        Offcard_Trans["Acti"] *= Batchsize # Multiply Batchsize
        Offcard_Trans["Acti"] *= self.Nlayer # Multiply Layer
        Offcard_Trans["Acti"] *= Iteration_Num # Multiply Iteration Number
        Offcard_Trans["Acti"] /= (mem_1k * mem_1k * mem_1k * mem_1k) # Scale down to TB
        Offcard_Trans["Acti_Round"] = self.Nlayer * 2 * Iteration_Num
        """
        return Compute, Memory, Arithmetic, Capacity, Offcard  # Offcard_Trans


class Sparse:
    COMP_PRF_PRJ = 1
    COMP_PRF_ACT = 1
    COMP_GNR_PRJ = 1
    COMP_GNR_ACT = 1
    MEM_PRF_PRJ = 1
    MEM_PRF_ACT = 1
    MEM_GNR_PRJ = 1
    MEM_GNR_ACT = 1
    Name = ""

    def __init__(self, input_sparse):
        self.Name = input_sparse["Name"]
        self.COMP_PRF_PRJ = input_sparse["COMP_PRF_PRJ"]
        self.COMP_PRF_ACT = input_sparse["COMP_PRF_ACT"]
        self.COMP_GNR_PRJ = input_sparse["COMP_GNR_PRJ"]
        self.COMP_GNR_ACT = input_sparse["COMP_GNR_ACT"]
        self.MEM_PRF_PRJ = input_sparse["MEM_PRF_PRJ"]
        self.MEM_PRF_ACT = input_sparse["MEM_PRF_ACT"]
        self.MEM_GNR_PRJ = input_sparse["MEM_GNR_PRJ"]
        self.MEM_GNR_ACT = input_sparse["MEM_GNR_ACT"]


class Link:
    def __init__(self, input_link):
        self.Name = input_link["Name"]
        self.Latency = input_link["Latency"]
        self.UniBW = input_link["UniBW"]
        self.BiBW = input_link["BiBW"]


class TransformerRoofline:
    # hardwares={"Homo":{}, "Hete":{}}
    hardwares = {}
    models = {}
    sparses = {}
    links = {}
    all_reduce_time = {}

    def __init__(
        self,
        hard_model_json="hardware_models.json",
        allreduce_config_file="allreduce_v100.xlsx",
        hardware_elements_json="hardware_elements.json",
    ):
        with open(hard_model_json, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            # print(type(data))
            # print(data)
            for hardware in data["hardware"]:
                name = hardware["Name"]
                # form_type = hardware["Type"]
                self.hardwares[name] = Hardware(hardware)
            for model in data["models"]:
                name = model["Name"]
                if "Wnum_in_MLP" not in model:
                    model["Wnum_in_MLP"] = 2
                self.models[name] = Model(model)
            for sparse in data["sparse"]:
                name = sparse["Name"]
                self.sparses[name] = Sparse(sparse)
            for link in data["links"]:
                name = link["Name"]
                self.links[name] = Link(link)
        wb = load_workbook(allreduce_config_file)
        for name in wb.sheetnames:
            sheet = wb[name]
            i = 3
            time_tmp = [0, 0, 0]
            size_loc = "B" + str(i)
            time_loc = "G" + str(i)
            while type(sheet[size_loc].value) == int:
                time_tmp.append(sheet[time_loc].value)
                i += 1
                size_loc = "B" + str(i)
                time_loc = "G" + str(i)
            self.all_reduce_time[name] = time_tmp
        # print(self.all_reduce_time)
        self.cost_model = CostEst(hardware_json=hardware_elements_json)

    def List_All_Hardware(self):
        # print("\033[1;31m%-20s%-20s%-20s\033[0m"%("Homo Hardware", "Peak TFLOPS", "BW(TB/s)"))
        # for key,value in self.hardwares["Homo"].items():
        # print("%-20s%-20d%-20d"%(value.Name, value.MM_TFLOPS, value.MM_BW_TBs ))
        print("\033[1;31m*******List_All_Hardware*******\033[0m")
        print(
            "\033[1;31m%-20s%-10s%-20s%-20s%-20s%-20s%-20s%-20s%-20s\033[0m"
            % (
                "Hardware",
                "Type",
                "MM_TFLOPS",
                "MM_BW(TB/s)",
                "MM_GP_TFLOPS",
                "MM_GP_BW(TB/s)",
                "MV_TFLOPS",
                "MV_BW(TB/s),",
                "Assign",
            )
        )
        for key, value in self.hardwares.items():
            print(
                "%-20s%-10s%-20d%-20.2f%-20d%-20.2f%-20d%-20.2f%-20s"
                % (
                    value.Name,
                    value.Form_Type,
                    value.MM_TFLOPS,
                    value.MM_BW_TBs,
                    value.MM_GP_TFLOPS,
                    value.MM_GP_BW_TBs,
                    value.MV_TFLOPS,
                    value.MV_BW_TBs,
                    value.Assign,
                )
            )

    def List_All_Model(self):
        print("\033[1;31m*******List_All_Model*******\033[0m")
        print(
            "\033[1;31m%-10s%-10s%-10s%-15s%-10s%-10s\033[0m"
            % ("Model", "Nhead", "Dmodel", "Dim_Per_Head", "Nlayer", "Max_Token")
        )
        for key, value in self.models.items():
            print(
                "%-10s%-10d%-10d%-15d%-10d%-10d"
                % (
                    value.Name,
                    value.Nhead,
                    value.Dmodel,
                    value.Dim_Per_Head,
                    value.Nlayer,
                    value.Max_Token,
                )
            )

    def List_Arithmetic(
        self, Ratio_P=[0.25], Batchsize=[32], Max_Token=[], Model=[], Sparse=[]
    ):
        print("\033[1;31m*******List_Arithmetic*******\033[0m")
        print(
            "\033[1;31m%-8s%-8s%-7s%-10s%-15s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s\033[0m"
            % (
                "Model",
                "Ratio_P",
                "Batch",
                "Max_Token",
                "Sparse",
                "Comp-P",
                "Mem-P",
                "Arith-P",
                "Comp-GP",
                "Mem-GP",
                "Arith-GP",
                "Comp-GA",
                "Mem-GA",
                "Arith-GA",
                "Comp-Ttl",
                "Mem-Ttl",
                "Arith-Ttl",
                "Weight Cap",
                "KV Cap",
                "Acti Cap",
            )
        )
        if Model == []:
            Model = [i for i in self.models.keys()]

        for model in Model:
            # for modelname,model in self.models.items():
            if Max_Token == []:
                Max_Token_List = [self.models[model].Max_Token]
            else:
                Max_Token_List = Max_Token
            Nlayer = self.models[model].Nlayer
            for ratiop in Ratio_P:
                for batch in Batchsize:
                    if Sparse == []:
                        Sparse = [i for i in self.sparses.keys()]
                    for sparse in Sparse:
                        for maxtoken in Max_Token_List:
                            Compute, Memory, Arithmetic, Capacity, _ = self.models[
                                model
                            ].Calculate_Arithmetic(
                                ratiop, batch, maxtoken, self.sparses[sparse]
                            )
                            print(
                                "%-8s%-8.2f%-7d%-10d%-15s%-10.1f%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f"
                                % (
                                    self.models[model].Name,
                                    ratiop,
                                    batch,
                                    maxtoken,
                                    sparse,
                                    Compute["PRF"]["PRJ"] + Compute["PRF"]["ACT"],
                                    Memory["PRF"]["PRJ"] + Memory["PRF"]["ACT"],
                                    Arithmetic["PRF"],
                                    Compute["GNR"]["PRJ"],
                                    Memory["GNR"]["PRJ"],
                                    Arithmetic["GNR"]["PRJ"],
                                    Compute["GNR"]["ACT"],
                                    Memory["GNR"]["ACT"],
                                    Arithmetic["GNR"]["ACT"],
                                    Compute["Total"],
                                    Memory["Total"],
                                    Arithmetic["Total"],
                                    Capacity["Weight"],
                                    Capacity["KVCache"],
                                    Capacity["Acti"],
                                )
                            )

    def Draw_Capacity(
        self, Ratio_P=[0.25], Batchsize=[32], Max_Token=[], Model=[], Sparse=[]
    ):
        print("\033[1;31m*******List_Capacity*******\033[0m")
        print(
            "\033[1;31m%-8s%-8s%-7s%-10s%-15s%-10s%-10s%-10s\033[0m"
            % (
                "Model",
                "Ratio_P",
                "Batch",
                "Max_Token",
                "Sparse",
                "Weight Cap",
                "KV Cap",
                "Acti Cap",
            )
        )
        if Model == []:
            Model = [i for i in self.models.keys()]

        Weight_List = []
        KVCache_List = []
        Acti_List = []
        title_legend = ""
        Label_List = []
        index = 0
        min_cap = 100
        for model in Model:
            model_legend = ""
            model_title = ""
            if len(Model) != 1:
                model_legend = model + "\n"
            else:
                model_title = model + " "
            Nlayer = self.models[model].Nlayer
            for ratiop in Ratio_P:
                ratiop_legend = ""
                ratiop_title = ""
                if len(Ratio_P) != 1:
                    ratiop_legend = "RP=" + str(ratiop) + "\n"
                else:
                    ratiop_title = "RP=" + str(ratiop) + " "
                for batch in Batchsize:
                    batchsize_legend = ""
                    batchsize_title = ""
                    if len(Batchsize) != 1:
                        batchsize_legend = "B=" + str(batch) + "\n"
                    else:
                        batchsize_title = "B=" + str(batch) + " "
                    if Sparse == []:
                        Sparse = [i for i in self.sparses.keys()]
                    for sparse in Sparse:
                        # title_legend += " S=" + sparse
                        sparse_legend = ""
                        sparse_title = ""
                        if len(Sparse) != 1:
                            sparse_legend = "S=" + str(sparse) + "\n"
                        else:
                            sparse_title = "S=" + str(sparse) + " "
                        if Max_Token == []:
                            Max_Token_List = [self.models[model].Max_Token]
                        else:
                            Max_Token_List = Max_Token
                        for maxtoken in Max_Token_List:
                            maxtoken_legend = ""
                            maxtoken_title = ""
                            if len(Max_Token) == 1:
                                maxtoken_title = "L=" + str(maxtoken) + "\n"
                            else:
                                maxtoken_legend = "L=" + str(maxtoken) + " "

                            _, _, _, Capacity, _ = self.models[
                                model
                            ].Calculate_Arithmetic(
                                ratiop, batch, maxtoken, self.sparses[sparse]
                            )
                            print(
                                "%-8s%-8.2f%-7d%-10d%-15s%-10.2f%-10.2f%-10.2f"
                                % (
                                    self.models[model].Name,
                                    ratiop,
                                    batch,
                                    maxtoken,
                                    sparse,  # Capacity["Weight"]*Nlayer, Capacity["KVCache"]*Nlayer, Capacity["Acti"]))
                                    Capacity["Weight"],
                                    Capacity["KVCache"],
                                    Capacity["Acti"],
                                )
                            )
                            Weight_List += [Capacity["Weight"], 0, 0, 0]
                            KVCache_List += [0, Capacity["KVCache"], 0, 0]
                            Acti_List += [0, 0, Capacity["Acti"], 0]
                            Label_List += [
                                (
                                    index,
                                    model_legend
                                    + ratiop_legend
                                    + batchsize_legend
                                    + sparse_legend
                                    + maxtoken_legend,
                                )
                            ]
                            title = (
                                model_title
                                + ratiop_title
                                + batchsize_title
                                + sparse_title
                                + maxtoken_title
                            )
                            index += 4
                            while min_cap > min(
                                [
                                    Capacity["Weight"],
                                    Capacity["KVCache"],
                                    Capacity["Acti"],
                                ]
                            ):
                                min_cap /= 10
            Weight_List += [0]
            KVCache_List += [0]
            Acti_List += [0]
            index += 1

        fontsize = 6 * 48 / len(Weight_List)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)

        plt.bar(np.arange(len(Weight_List)), Weight_List, label="Weight", log="True")
        plt.bar(np.arange(len(KVCache_List)), KVCache_List, label="KVCache", log="True")
        plt.bar(np.arange(len(Acti_List)), Acti_List, label="Acti", log="True")

        for loc, text in Label_List:
            plt.text(
                x=loc,
                y=min_cap,
                s=text,
                fontsize=fontsize,
            )

        plt.title("Capacity Requirement(GB) " + title)
        plt.legend()
        plt.show()

    def Compute_Timebreakdown_Iteration(
        self,
        Prompt_Len,
        Step,
        Batchsize,
        Model,
        Hardware,
        Sparse="default",
        Pipeline_Stage=1,
    ) -> (float, float):
        ### Output dict: out[str_model_name][str_hardware_name][ratiop][batch][maxtoken]={"Compute":{...}, "Memory":{...}, "Arithmetic":{...}, "Timebreakdown":{...}}

        hard = Hardware
        model = Model
        Stage = ""
        if Step == 0:
            Stage = "Prefill"
        else:
            Stage = "Generation"
        batch = Batchsize
        sparse = Sparse

        MM_TFLOPS = self.hardwares[hard].MM_TFLOPS
        MM_GP_TFLOPS = self.hardwares[hard].MM_GP_TFLOPS
        MV_TFLOPS = self.hardwares[hard].MV_TFLOPS
        MM_BW_TBs = self.hardwares[hard].MM_BW_TBs
        MM_GP_BW_TBs = self.hardwares[hard].MM_GP_BW_TBs
        MV_BW_TBs = self.hardwares[hard].MV_BW_TBs
        MM_Card_Num = int(self.hardwares[hard].MM_Card_Num)
        MM_GP_Card_Num = int(self.hardwares[hard].MM_GP_Card_Num)
        MV_Card_Num = self.hardwares[hard].MV_Card_Num
        MM_Sweet_Point = self.hardwares[hard].MM_Sweet_Point
        MM_GP_Sweet_Point = self.hardwares[hard].MM_GP_Sweet_Point
        MV_Sweet_Point = self.hardwares[hard].MV_Sweet_Point
        Assign = self.hardwares[hard].Assign

        MM_Card_Per_Stage = int(MM_Card_Num / Pipeline_Stage)
        MM_GP_Card_Per_Stage = int(MM_GP_Card_Num / Pipeline_Stage)
        MV_Card_Per_Stage = int(MV_Card_Num / Pipeline_Stage)

        MOE_Quantity = int(self.models[model].MOE_Quantity)
        MOE_Activate = int(self.models[model].MOE_Activate)

        MM_Average_Max = MOE_Activate
        MM_GP_Average_Max = MOE_Activate
        if MOE_Quantity != MOE_Activate:
            MM_Average_Max = self.MOE_MC(
                Total_Expert=MOE_Quantity,
                Active_Expert=MOE_Activate,
                Card_Number=MM_Card_Per_Stage,
                P_List=0,
                Batch=batch,
                Sample=100000,
            )
            MM_GP_Average_Max = self.MOE_MC(
                Total_Expert=MOE_Quantity,
                Active_Expert=MOE_Activate,
                Card_Number=MM_GP_Card_Per_Stage,
                P_List=0,
                Batch=batch,
                Sample=100000,
            )

        maxtoken_info = {}
        # print("hardware:%s, model:%s, ratiop:%.2f, batch:%d, sparse:%s, maxtoken:%d"%(hard, model, ratiop, batch, sparse, maxtoken))

        Compute, Memory, Arithmetic, _, Offcard_Trans = self.models[
            model
        ].Calculate_Arithmetic_Iteration(
            Prompt_Len, Step, batch, self.sparses[sparse], Pipeline_Stage, Assign
        )

        Real_Perf = {"PRF": 0, "GNR": {"PRJ": 0, "ACT": 0}, "PRF-PRJ": 0, "PRF-ACT": 0}
        Timebreakdown = {
            "PRF": 0,
            "GNR": {"PRJ": 0, "ACT": 0},
            "PRF-PRJ": 0,
            "PRF-ACT": 0,
            "PRF-Shared_PRJ": 0,
            "PRF-FFN_PRJ": 0,
            "GNR-Shared_PRJ": 0,
            "GNR-FFN_PRJ": 0,
            "OFFCARD": {
                "PRF": {"inter_Stage": 0, "intra_Stage": 0, "out_Card": 0, "Total": 0},
                "GNR-PRJ": {
                    "inter_Stage": 0,
                    "intra_Stage": 0,
                    "out_Card": 0,
                    "Total": 0,
                },
                "GNR-ACT": {
                    "inter_Stage": 0,
                    "intra_Stage": 0,
                    "out_Card": 0,
                    "Total": 0,
                },
                "Total": 0,
            },
            "Total": 0,
        }
        Util = {"PRF": 0, "GNR": {"PRJ": 0, "ACT": 0}, "Total": 0}

        if Step == 0:
            Real_Perf["PRF"] = (
                MM_TFLOPS
                if Arithmetic["PRF"] > MM_Sweet_Point
                else Arithmetic["PRF"] * MM_BW_TBs
            )
            Real_Perf["PRF-PRJ"] = (
                MM_TFLOPS
                if Arithmetic["PRF-PRJ"] > MM_Sweet_Point
                else Arithmetic["PRF-PRJ"] * MM_BW_TBs
            )
            Real_Perf["PRF-ACT"] = (
                MM_TFLOPS
                if Arithmetic["PRF-ACT"] > MM_Sweet_Point
                else Arithmetic["PRF-ACT"] * MM_BW_TBs
            )
            Real_Perf["PRF-Shared_PRJ"] = (
                MM_TFLOPS
                if Arithmetic["PRF-Shared_PRJ"] > MM_Sweet_Point
                else Arithmetic["PRF-Shared_PRJ"] * MM_BW_TBs
            )
            Real_Perf["PRF-FFN_PRJ"] = (
                MM_TFLOPS
                if Arithmetic["PRF-FFN_PRJ"] > MM_Sweet_Point
                else Arithmetic["PRF-FFN_PRJ"] * MM_BW_TBs
            )
        if Step > 0:
            Real_Perf["GNR-Shared_PRJ"] = (
                MM_GP_TFLOPS
                if Arithmetic["GNR-Shared_PRJ"] > MM_Sweet_Point
                else Arithmetic["GNR-Shared_PRJ"] * MM_BW_TBs
            )
            Real_Perf["GNR-FFN_PRJ"] = (
                MM_GP_TFLOPS
                if Arithmetic["GNR-FFN_PRJ"] > MM_Sweet_Point
                else Arithmetic["GNR-FFN_PRJ"] * MM_BW_TBs
            )
            # print(f'{Arithmetic["GNR-FFN_PRJ"]=} {MM_Sweet_Point=}')
            Real_Perf["GNR"]["PRJ"] = (
                MM_GP_TFLOPS
                if Arithmetic["GNR"]["PRJ"] > MM_GP_Sweet_Point
                else Arithmetic["GNR"]["PRJ"] * MM_GP_BW_TBs
            )
            Real_Perf["GNR"]["ACT"] = (
                MV_TFLOPS
                if Arithmetic["GNR"]["ACT"] > MV_Sweet_Point
                else Arithmetic["GNR"]["ACT"] * MV_BW_TBs
            )

        if Step == 0:
            PRF_Shared_PRJ_Time_Per_Stage = (
                Compute["PRF"]["Shared_PRJ"] / Pipeline_Stage
            ) / (Real_Perf["PRF-Shared_PRJ"] / Pipeline_Stage)
            Timebreakdown["PRF-Shared_PRJ"] = (
                PRF_Shared_PRJ_Time_Per_Stage * Pipeline_Stage
            )
            PRF_FFN_PRJ_Time_Per_Stage = (
                Compute["PRF"]["FFN_PRJ"] / Pipeline_Stage
            ) / (Real_Perf["PRF-FFN_PRJ"] / Pipeline_Stage)
            Timebreakdown["PRF-FFN_PRJ"] = (
                PRF_FFN_PRJ_Time_Per_Stage
                * Pipeline_Stage
                / MOE_Activate
                * MM_Average_Max
            )
            # PRF_PRJ_Time_Per_Stage = ((Compute["PRF"]["PRJ"]) / Pipeline_Stage) / (Real_Perf["PRF-PRJ"] / Pipeline_Stage)
            # Timebreakdown["PRF-PRJ"] = PRF_PRJ_Time_Per_Stage * Pipeline_Stage
            Timebreakdown["PRF-PRJ"] = (
                Timebreakdown["PRF-Shared_PRJ"] + Timebreakdown["PRF-FFN_PRJ"]
            )
            PRF_ACT_Time_Per_Stage = ((Compute["PRF"]["ACT"]) / Pipeline_Stage) / (
                Real_Perf["PRF-ACT"] / Pipeline_Stage
            )
            Timebreakdown["PRF-ACT"] = PRF_ACT_Time_Per_Stage * Pipeline_Stage
            Timebreakdown["PRF"] = Timebreakdown["PRF-PRJ"] + Timebreakdown["PRF-ACT"]
        if Step > 0:
            GNR_Shared_PRJ_Time_Per_Stage = (
                Compute["GNR"]["Shared_PRJ"] / Pipeline_Stage
            ) / (Real_Perf["GNR-Shared_PRJ"] / Pipeline_Stage)
            # print(f'{Real_Perf["GNR-Shared_PRJ"]=}')
            Timebreakdown["GNR-Shared_PRJ"] = (
                GNR_Shared_PRJ_Time_Per_Stage * Pipeline_Stage
            )
            GNR_FFN_PRJ_Time_Per_Stage = (
                Compute["GNR"]["FFN_PRJ"] / Pipeline_Stage
            ) / (Real_Perf["GNR-FFN_PRJ"] / Pipeline_Stage)
            # print(f'{Real_Perf["GNR-FFN_PRJ"]=}')
            Timebreakdown["GNR-FFN_PRJ"] = (
                GNR_FFN_PRJ_Time_Per_Stage
                * Pipeline_Stage
                / MOE_Activate
                * MM_GP_Average_Max
            )
            # GNR_PRJ_Time_Per_Stage = (Compute["GNR"]["PRJ"] / Pipeline_Stage) / (Real_Perf["GNR"]["PRJ"] / Pipeline_Stage)
            # Timebreakdown["GNR"]["PRJ"] = GNR_PRJ_Time_Per_Stage * Pipeline_Stage / MOE_Activate * MM_GP_Average_Max
            Timebreakdown["GNR"]["PRJ"] = (
                Timebreakdown["GNR-Shared_PRJ"] + Timebreakdown["GNR-FFN_PRJ"]
            )
            GNR_ACT_Time_Per_Stage = (Compute["GNR"]["ACT"] / Pipeline_Stage) / (
                Real_Perf["GNR"]["ACT"] / Pipeline_Stage
            )
            # print(f'{Real_Perf["GNR"]["ACT"]=}')
            Timebreakdown["GNR"]["ACT"] = GNR_ACT_Time_Per_Stage * Pipeline_Stage
            # print(f"{GNR_Shared_PRJ_Time_Per_Stage=}")
            # print(f"{GNR_FFN_PRJ_Time_Per_Stage=}")
            # print(f"{GNR_ACT_Time_Per_Stage=} {Pipeline_Stage=}")

        if MM_Card_Per_Stage < 1 or MM_GP_Card_Per_Stage < 1 or MV_Card_Per_Stage < 1:
            raise ValueError(
                "Pipeline stage number must be larger than hardware chip number!"
            )

        if Step == 0:
            Timebreakdown["OFFCARD"]["PRF"]["inter_Stage"] += (
                self.Get_Offcard_Time(Offcard_Trans["PRF"]["inter_Stage"], 2) / 1e6
            )
            Offcard_Trans["PRF"]["out_Card"][0] /= MM_Card_Per_Stage
            Timebreakdown["OFFCARD"]["PRF"]["out_Card"] += (
                self.Get_Offcard_Time(Offcard_Trans["PRF"]["out_Card"], 2) / 1e6
            )
            if MM_Card_Per_Stage > 1:  # model parallelism exists in each pipeline stage
                Timebreakdown["OFFCARD"]["PRF"]["intra_Stage"] += (
                    self.Get_Offcard_Time(
                        Offcard_Trans["PRF"]["intra_Stage"], MM_Card_Per_Stage
                    )
                    / 1e6
                )
            Timebreakdown["OFFCARD"]["PRF"]["Total"] = (
                Timebreakdown["OFFCARD"]["PRF"]["inter_Stage"]
                + Timebreakdown["OFFCARD"]["PRF"]["intra_Stage"]
                + Timebreakdown["OFFCARD"]["PRF"]["out_Card"]
            )

        if Step > 0:
            Timebreakdown["OFFCARD"]["GNR-PRJ"]["inter_Stage"] += (
                self.Get_Offcard_Time(Offcard_Trans["GNR-PRJ"]["inter_Stage"], 2) / 1e6
            )
            Timebreakdown["OFFCARD"]["GNR-ACT"]["inter_Stage"] += (
                self.Get_Offcard_Time(Offcard_Trans["GNR-ACT"]["inter_Stage"], 2) / 1e6
            )

            Offcard_Trans["GNR-PRJ"]["out_Card"][0] /= MM_GP_Card_Per_Stage
            Timebreakdown["OFFCARD"]["GNR-PRJ"]["out_Card"] += (
                self.Get_Offcard_Time(Offcard_Trans["GNR-PRJ"]["out_Card"], 2) / 1e6
            )
            Offcard_Trans["GNR-ACT"]["out_Card"][0] /= MV_Card_Per_Stage
            Timebreakdown["OFFCARD"]["GNR-ACT"]["out_Card"] += (
                self.Get_Offcard_Time(Offcard_Trans["GNR-ACT"]["out_Card"], 2) / 1e6
            )

            if MM_GP_Card_Per_Stage > 1:
                Timebreakdown["OFFCARD"]["GNR-PRJ"]["intra_Stage"] += (
                    self.Get_Offcard_Time(
                        Offcard_Trans["GNR-PRJ"]["intra_Stage"], MM_GP_Card_Per_Stage
                    )
                    / 1e6
                )
            if MV_Card_Per_Stage > 1:
                Timebreakdown["OFFCARD"]["GNR-ACT"]["intra_Stage"] += (
                    self.Get_Offcard_Time(
                        Offcard_Trans["GNR-ACT"]["intra_Stage"], MV_Card_Per_Stage
                    )
                    / 1e6
                )

            if self.models[model].Parallel_FFN:
                Timebreakdown["OFFCARD"]["GNR-PRJ"]["Total"] = (
                    Timebreakdown["OFFCARD"]["GNR-PRJ"]["inter_Stage"]
                    + Timebreakdown["OFFCARD"]["GNR-PRJ"]["intra_Stage"] / 2
                    + Timebreakdown["OFFCARD"]["GNR-PRJ"]["out_Card"]
                )
            else:
                Timebreakdown["OFFCARD"]["GNR-PRJ"]["Total"] = (
                    Timebreakdown["OFFCARD"]["GNR-PRJ"]["inter_Stage"]
                    + Timebreakdown["OFFCARD"]["GNR-PRJ"]["intra_Stage"]
                    + Timebreakdown["OFFCARD"]["GNR-PRJ"]["out_Card"]
                )
            Timebreakdown["OFFCARD"]["GNR-ACT"]["Total"] = (
                Timebreakdown["OFFCARD"]["GNR-ACT"]["inter_Stage"]
                + Timebreakdown["OFFCARD"]["GNR-ACT"]["intra_Stage"]
                + Timebreakdown["OFFCARD"]["GNR-ACT"]["out_Card"]
            )
        Timebreakdown["OFFCARD"]["Total"] = (
            Timebreakdown["OFFCARD"]["PRF"]["Total"]
            + Timebreakdown["OFFCARD"]["GNR-PRJ"]["Total"]
            + Timebreakdown["OFFCARD"]["GNR-ACT"]["Total"]
        )
        Timebreakdown["Total"] = (
            Timebreakdown["PRF"]
            + Timebreakdown["GNR"]["PRJ"]
            + Timebreakdown["GNR"]["ACT"]
            + Timebreakdown["OFFCARD"]["Total"]
        )

        if Step == 0:
            Util["PRF"] = (
                (Compute["PRF"]["PRJ"] + Compute["PRF"]["ACT"])
                * Pipeline_Stage
                / (Timebreakdown["PRF"] + Timebreakdown["OFFCARD"]["PRF"]["Total"])
                / MM_TFLOPS
            )
        if Step > 0:
            Util["GNR"]["PRJ"] = (
                Compute["GNR"]["PRJ"]
                * Pipeline_Stage
                / (
                    Timebreakdown["GNR"]["PRJ"]
                    + Timebreakdown["OFFCARD"]["GNR-PRJ"]["Total"]
                )
                / MM_GP_TFLOPS
            )
            Util["GNR"]["ACT"] = (
                Compute["GNR"]["ACT"]
                * Pipeline_Stage
                / (
                    Timebreakdown["GNR"]["ACT"]
                    + Timebreakdown["OFFCARD"]["GNR-ACT"]["Total"]
                )
                / MV_TFLOPS
            )
            if self.hardwares[hard].Form_Type == "Homo":
                Util["GNR"]["Total"] = (
                    (Compute["GNR"]["PRJ"] + Compute["GNR"]["ACT"])
                    * Pipeline_Stage
                    / (
                        Timebreakdown["Total"]
                        - Timebreakdown["PRF"]
                        - Timebreakdown["OFFCARD"]["PRF"]["Total"]
                    )
                    / MM_GP_TFLOPS
                )
        Util["Total"] = (
            Compute["Total"] * Pipeline_Stage / Timebreakdown["Total"] / MM_TFLOPS
        )

        maxtoken_info["Compute"] = Compute
        maxtoken_info["Memory"] = Memory
        maxtoken_info["Offcard"] = Offcard_Trans
        maxtoken_info["Arithmetic"] = Arithmetic
        maxtoken_info["Real_Perf"] = Real_Perf
        maxtoken_info["Timebreakdown"] = Timebreakdown
        maxtoken_info["Util"] = Util

        if Step == 0:
            return (
                Timebreakdown["PRF-PRJ"] + Timebreakdown["OFFCARD"]["PRF"]["Total"],
                Timebreakdown["PRF-ACT"],
            )
        if Step > 0:
            # print(
            #     f'{Timebreakdown["GNR"]["PRJ"]=} \n\
            #    {Timebreakdown["OFFCARD"]["GNR-PRJ"]["Total"]=} \n\
            #     {Timebreakdown["GNR"]["ACT"]=} \n\
            #     {Timebreakdown["OFFCARD"]["GNR-ACT"]["Total"]=}'
            # )
            return (
                Timebreakdown["GNR"]["PRJ"]
                + Timebreakdown["OFFCARD"]["GNR-PRJ"]["Total"],
                Timebreakdown["GNR"]["ACT"]
                + Timebreakdown["OFFCARD"]["GNR-ACT"]["Total"],
            )

    def Generate_Trace(
        self,
        Prompt_Len,
        Generation_Len,
        Batchsize,
        Model,
        Hardware,
        Sparse="default",
        Pipeline_Stage=1,
    ):
        trace = []
        model = self.models[Model]
        hard = self.hardwares[Hardware]
        assert hard.Form_Type == "Homo"
        card_num = hard.MM_Card_Num
        MM_TFLOPS = hard.MM_TFLOPS / card_num
        MM_BW = hard.MM_BW_TBs / card_num
        MV_TFLOPS = hard.MV_TFLOPS / card_num
        MV_BW = hard.MV_BW_TBs / card_num
        names = [
            "QKV",
            "Q*KT",
            "Softmax",
            "Score*V",
            "W0",
            "AllReduce",
            "FFN1",
            "FFN2",
            "AllReduce",
        ]
        for iter_idx in range(Generation_Len):
            stage = "Prefill" if iter_idx == 0 else "Generation"
            for layer in range(model.Nlayer):
                for name in names:
                    op = {}
                    op["name"] = name + "_Iter_%d_Layer_%d" % (iter_idx, layer)
                    if name == "QKV":
                        op["type"] = "Gemm"
                        m = Batchsize * Prompt_Len if stage == "Prefill" else Batchsize
                        compute = (
                            m
                            * model.Dmodel
                            / card_num
                            * model.Dmodel
                            * 2
                            * 3
                            / (comp_1k * comp_1k * comp_1k * comp_1k)
                        )
                        memory = (
                            model.Dmodel
                            / card_num
                            * model.Dmodel
                            * 2
                            * 3
                            / (mem_1k * mem_1k * mem_1k * mem_1k)
                        )
                        execution_time = max(compute / MM_TFLOPS, memory / MM_BW)
                        compute_util = compute / (execution_time * MM_TFLOPS)
                        memory_util = memory / (execution_time * MM_BW)
                        op["shape"] = "m=%d, n=%d, k=%d" % (
                            m,
                            model.Dmodel / card_num * 3,
                            model.Dmodel,
                        )
                        pass
                    elif name == "Q*KT":
                        op["type"] = "BatchedGemm"
                        m = Prompt_Len if stage == "Prefill" else 1
                        batch = Batchsize * model.Nhead / card_num
                        compute = (
                            m
                            * (Prompt_Len + iter_idx)
                            * model.Dim_Per_Head
                            * batch
                            * 2
                            / (comp_1k * comp_1k * comp_1k * comp_1k)
                        )
                        memory = (
                            (Prompt_Len + iter_idx)
                            * model.Dim_Per_Head
                            * batch
                            * 2
                            / (mem_1k * mem_1k * mem_1k * mem_1k)
                        )
                        execution_time = max(compute / MV_TFLOPS, memory / MV_BW)
                        compute_util = compute / (execution_time * MV_TFLOPS)
                        memory_util = memory / (execution_time * MV_BW)
                        op["shape"] = "m=%d, n=%d, k=%d" % (
                            m,
                            Prompt_Len + iter_idx,
                            model.Dim_Per_Head,
                        )
                        op["batch"] = batch
                        pass
                    elif name == "Softmax":
                        op["type"] = "BatchedElement"
                        m = Prompt_Len if stage == "Prefill" else 1
                        compute = 0
                        memory = 0
                        execution_time = 0
                        compute_util = 0
                        memory_util = 0
                        op["shape"] = "m=%d, n=%d" % (m, Prompt_Len + iter_idx)
                        op["batch"] = Batchsize * model.Nhead / card_num
                        pass
                    elif name == "Score*V":
                        op["type"] = "BatchedGemm"
                        m = Prompt_Len if stage == "Prefill" else 1
                        batch = Batchsize * model.Nhead / card_num
                        compute = (
                            m
                            * (Prompt_Len + iter_idx)
                            * model.Dim_Per_Head
                            * batch
                            * 2
                            / (comp_1k * comp_1k * comp_1k * comp_1k)
                        )
                        memory = (
                            (Prompt_Len + iter_idx)
                            * model.Dim_Per_Head
                            * batch
                            * 2
                            / (mem_1k * mem_1k * mem_1k * mem_1k)
                        )
                        execution_time = max(compute / MV_TFLOPS, memory / MV_BW)
                        compute_util = compute / (execution_time * MV_TFLOPS)
                        memory_util = memory / (execution_time * MV_BW)
                        op["shape"] = "m=%d, n=%d, k=%d" % (
                            m,
                            Prompt_Len + iter_idx,
                            model.Dim_Per_Head,
                        )
                        op["batch"] = batch
                        pass
                    elif name == "W0":
                        op["type"] = "Gemm"
                        m = Batchsize * Prompt_Len if stage == "Prefill" else Batchsize
                        compute = (
                            m
                            * model.Dmodel
                            * model.Dmodel
                            / card_num
                            * 2
                            / (comp_1k * comp_1k * comp_1k * comp_1k)
                        )
                        memory = (
                            model.Dmodel
                            * model.Dmodel
                            / card_num
                            * 2
                            / (mem_1k * mem_1k * mem_1k * mem_1k)
                        )
                        execution_time = max(compute / MM_TFLOPS, memory / MM_BW)
                        compute_util = compute / (execution_time * MM_TFLOPS)
                        memory_util = memory / (execution_time * MM_BW)
                        op["shape"] = "m=%d, n=%d, k=%d" % (
                            m,
                            model.Dmodel,
                            model.Dmodel / card_num,
                        )
                        pass
                    elif name == "AllReduce":
                        op["type"] = "Communication"
                        m = Batchsize * Prompt_Len if stage == "Prefill" else Batchsize
                        op["shape"] = "m=%d, n=%d" % (m, model.Dmodel)
                        op["card_num"] = card_num
                        compute = 0
                        memory = 0
                        compute_util = 0
                        memory_util = 0
                        execution_time = (
                            self.Get_Offcard_Time((m * model.Dmodel * 2, 1), card_num)
                            / 1e6
                        )
                        pass
                    elif name == "FFN1":
                        op["type"] = "Gemm"
                        m = Batchsize * Prompt_Len if stage == "Prefill" else Batchsize
                        compute = (
                            m
                            * model.FFN_Hidden
                            / card_num
                            * model.Dmodel
                            * 2
                            / (comp_1k * comp_1k * comp_1k * comp_1k)
                        )
                        memory = (
                            model.FFN_Hidden
                            / card_num
                            * model.Dmodel
                            * 2
                            / (mem_1k * mem_1k * mem_1k * mem_1k)
                        )
                        execution_time = max(compute / MM_TFLOPS, memory / MM_BW)
                        compute_util = compute / (execution_time * MM_TFLOPS)
                        memory_util = memory / (execution_time * MM_BW)
                        op["shape"] = "m=%d, n=%d, k=%d" % (
                            m,
                            model.FFN_Hidden / card_num,
                            model.Dmodel,
                        )
                        pass
                    elif name == "FFN2":
                        op["type"] = "Gemm"
                        m = Batchsize * Prompt_Len if stage == "Prefill" else Batchsize
                        compute = (
                            m
                            * model.Dmodel
                            * model.FFN_Hidden
                            / card_num
                            * 2
                            / (comp_1k * comp_1k * comp_1k * comp_1k)
                        )
                        memory = (
                            model.Dmodel
                            * model.FFN_Hidden
                            / card_num
                            * 2
                            / (mem_1k * mem_1k * mem_1k * mem_1k)
                        )
                        execution_time = max(compute / MM_TFLOPS, memory / MM_BW)
                        compute_util = compute / (execution_time * MM_TFLOPS)
                        memory_util = memory / (execution_time * MM_BW)
                        op["shape"] = "m=%d, n=%d, k=%d" % (
                            m,
                            model.Dmodel,
                            model.FFN_Hidden / card_num,
                        )
                        pass
                    else:
                        raise ValueError("Unsupported OP name!")
                    op["stage"] = stage
                    op["compute"] = compute
                    op["memory"] = memory
                    op["latency"] = execution_time
                    op["compute_util"] = compute_util
                    op["memory_util"] = memory_util

                    trace.append(op)
        return trace

    def Compute_Timebreakdown(
        self,
        Ratio_P=[0.25],
        Batchsize=[32],
        Max_Token=[],
        Model=[],
        Hardware=[],
        Sparse=[],
        Pipeline_Stage=1,
    ):
        ### Output dict: out[str_model_name][str_hardware_name][ratiop][batch][maxtoken]={"Compute":{...}, "Memory":{...}, "Arithmetic":{...}, "Timebreakdown":{...}}
        out = {}
        if Model == []:
            Model = [i for i in self.models.keys()]
        if Hardware == []:
            Hardware = [i for i in self.hardwares.keys()]
        if Sparse == []:
            Sparse = [i for i in self.sparses.keys()]

        for hard in Hardware:
            # print(hard)
            if hard == None:
                continue
            hard_info = {}
            MM_TFLOPS = self.hardwares[hard].MM_TFLOPS
            MM_GP_TFLOPS = self.hardwares[hard].MM_GP_TFLOPS
            MV_TFLOPS = self.hardwares[hard].MV_TFLOPS
            MM_BW_TBs = self.hardwares[hard].MM_BW_TBs
            MM_GP_BW_TBs = self.hardwares[hard].MM_GP_BW_TBs
            MV_BW_TBs = self.hardwares[hard].MV_BW_TBs
            MM_Card_Num = int(self.hardwares[hard].MM_Card_Num)
            MM_GP_Card_Num = int(self.hardwares[hard].MM_GP_Card_Num)
            MV_Card_Num = self.hardwares[hard].MV_Card_Num
            MM_Sweet_Point = self.hardwares[hard].MM_Sweet_Point
            MM_GP_Sweet_Point = self.hardwares[hard].MM_GP_Sweet_Point
            MV_Sweet_Point = self.hardwares[hard].MV_Sweet_Point
            Offcard_Type = self.hardwares[hard].Offcard_Type
            Assign = self.hardwares[hard].Assign
            AllReduce_Links = self.hardwares[hard].AllReduce_Links

            MM_Card_Per_Stage = int(MM_Card_Num / Pipeline_Stage)
            MM_GP_Card_Per_Stage = int(MM_GP_Card_Num / Pipeline_Stage)
            MV_Card_Per_Stage = int(MV_Card_Num / Pipeline_Stage)
            for model in Model:
                model_info = {}
                MOE_Quantity = int(self.models[model].MOE_Quantity)
                MOE_Activate = int(self.models[model].MOE_Activate)
                for ratiop in Ratio_P:
                    ratio_info = {}
                    for batch in Batchsize:
                        batch_info = {}
                        batch_name = batch
                        MM_Average_Max = MOE_Activate
                        MM_GP_Average_Max = MOE_Activate
                        if MOE_Quantity != MOE_Activate:
                            MM_Average_Max = self.MOE_MC(
                                Total_Expert=MOE_Quantity,
                                Active_Expert=MOE_Activate,
                                Card_Number=MM_Card_Per_Stage,
                                P_List=0,
                                Batch=batch,
                                Sample=100000,
                            )
                            MM_GP_Average_Max = self.MOE_MC(
                                Total_Expert=MOE_Quantity,
                                Active_Expert=MOE_Activate,
                                Card_Number=MM_GP_Card_Per_Stage,
                                P_List=0,
                                Batch=batch,
                                Sample=100000,
                            )
                        for sparse in Sparse:
                            sparse_info = {}
                            if Max_Token == []:
                                Max_Token_List = [self.models[model].Max_Token]
                            else:
                                Max_Token_List = Max_Token
                            for maxtoken in Max_Token_List:
                                maxtoken_info = {}
                                if batch_name in ["Max", "max", "MAX"]:
                                    batch = self.Get_Max_Batch_by_Capacity(
                                        hard, model, ratiop, maxtoken, sparse
                                    )
                                    batch = self.Find_Max_Power_2(1, batch)
                                maxtoken_info["batchsize"] = batch
                                # print("hardware:%s, model:%s, ratiop:%.2f, batch:%d, sparse:%s, maxtoken:%d"%(hard, model, ratiop, batch, sparse, maxtoken))

                                (
                                    Compute,
                                    Memory,
                                    Arithmetic,
                                    _,
                                    Offcard_Trans,
                                ) = self.models[model].Calculate_Arithmetic(
                                    ratiop,
                                    batch,
                                    maxtoken,
                                    self.sparses[sparse],
                                    Pipeline_Stage,
                                    Assign,
                                )

                                Real_Perf = {
                                    "PRF": 0,
                                    "GNR": {"PRJ": 0, "ACT": 0},
                                    "PRF-PRJ": 0,
                                    "PRF-ACT": 0,
                                }
                                Timebreakdown = {
                                    "PRF": 0,
                                    "GNR": {"PRJ": 0, "ACT": 0},
                                    "PRF-PRJ": 0,
                                    "PRF-ACT": 0,
                                    "OFFCARD": {
                                        "PRF": {
                                            "inter_Stage": 0,
                                            "intra_Stage": 0,
                                            "out_Card": 0,
                                            "Total": 0,
                                            "AllReduce": {},
                                        },
                                        "GNR-PRJ": {
                                            "inter_Stage": 0,
                                            "intra_Stage": 0,
                                            "out_Card": 0,
                                            "Total": 0,
                                            "AllReduce": {},
                                        },
                                        "GNR-ACT": {
                                            "inter_Stage": 0,
                                            "intra_Stage": 0,
                                            "out_Card": 0,
                                            "Total": 0,
                                            "AllReduce": {},
                                        },
                                        "Total": 0,
                                    },
                                    "Total": 0,
                                }
                                Util = {
                                    "PRF": 0,
                                    "GNR": {"PRJ": 0, "ACT": 0},
                                    "Total": 0,
                                }
                                Throughput = 0

                                Real_Perf["PRF"] = (
                                    MM_TFLOPS
                                    if Arithmetic["PRF"] > MM_Sweet_Point
                                    else Arithmetic["PRF"] * MM_BW_TBs
                                )
                                Real_Perf["PRF-PRJ"] = (
                                    MM_TFLOPS
                                    if Arithmetic["PRF-PRJ"] > MM_Sweet_Point
                                    else Arithmetic["PRF-PRJ"] * MM_BW_TBs
                                )
                                Real_Perf["PRF-ACT"] = (
                                    MM_TFLOPS
                                    if Arithmetic["PRF-ACT"] > MM_Sweet_Point
                                    else Arithmetic["PRF-ACT"] * MM_BW_TBs
                                )
                                Real_Perf["PRF-Shared_PRJ"] = (
                                    MM_TFLOPS
                                    if Arithmetic["PRF-Shared_PRJ"] > MM_Sweet_Point
                                    else Arithmetic["PRF-Shared_PRJ"] * MM_BW_TBs
                                )
                                Real_Perf["PRF-FFN_PRJ"] = (
                                    MM_TFLOPS
                                    if Arithmetic["PRF-FFN_PRJ"] > MM_Sweet_Point
                                    else Arithmetic["PRF-FFN_PRJ"] * MM_BW_TBs
                                )
                                Real_Perf["GNR-Shared_PRJ"] = (
                                    MM_GP_TFLOPS
                                    if Arithmetic["GNR-Shared_PRJ"] > MM_Sweet_Point
                                    else Arithmetic["GNR-Shared_PRJ"] * MM_BW_TBs
                                )
                                Real_Perf["GNR-FFN_PRJ"] = (
                                    MM_GP_TFLOPS
                                    if Arithmetic["GNR-FFN_PRJ"] > MM_Sweet_Point
                                    else Arithmetic["GNR-FFN_PRJ"] * MM_BW_TBs
                                )
                                Real_Perf["GNR"]["PRJ"] = (
                                    MM_GP_TFLOPS
                                    if Arithmetic["GNR"]["PRJ"] > MM_GP_Sweet_Point
                                    else Arithmetic["GNR"]["PRJ"] * MM_GP_BW_TBs
                                )
                                Real_Perf["GNR"]["ACT"] = (
                                    MV_TFLOPS
                                    if Arithmetic["GNR"]["ACT"] > MV_Sweet_Point
                                    else Arithmetic["GNR"]["ACT"] * MV_BW_TBs
                                )

                                PRF_Shared_PRJ_Time_Per_Stage = (
                                    Compute["PRF"]["Shared_PRJ"] / Pipeline_Stage
                                ) / (Real_Perf["PRF-Shared_PRJ"] / Pipeline_Stage)
                                Timebreakdown["PRF-Shared_PRJ"] = (
                                    PRF_Shared_PRJ_Time_Per_Stage * Pipeline_Stage
                                )
                                PRF_FFN_PRJ_Time_Per_Stage = (
                                    Compute["PRF"]["FFN_PRJ"] / Pipeline_Stage
                                ) / (Real_Perf["PRF-FFN_PRJ"] / Pipeline_Stage)
                                Timebreakdown["PRF-FFN_PRJ"] = (
                                    PRF_FFN_PRJ_Time_Per_Stage
                                    * Pipeline_Stage
                                    / MOE_Activate
                                    * MM_Average_Max
                                )
                                # PRF_PRJ_Time_Per_Stage = ((Compute["PRF"]["PRJ"]) / Pipeline_Stage) / (Real_Perf["PRF-PRJ"] / Pipeline_Stage)
                                # Timebreakdown["PRF-PRJ"] = PRF_PRJ_Time_Per_Stage * Pipeline_Stage
                                Timebreakdown["PRF-PRJ"] = (
                                    Timebreakdown["PRF-Shared_PRJ"]
                                    + Timebreakdown["PRF-FFN_PRJ"]
                                )
                                PRF_ACT_Time_Per_Stage = (
                                    (Compute["PRF"]["ACT"]) / Pipeline_Stage
                                ) / (Real_Perf["PRF-ACT"] / Pipeline_Stage)
                                Timebreakdown["PRF-ACT"] = (
                                    PRF_ACT_Time_Per_Stage * Pipeline_Stage
                                )
                                Timebreakdown["PRF"] = (
                                    Timebreakdown["PRF-PRJ"] + Timebreakdown["PRF-ACT"]
                                )

                                GNR_Shared_PRJ_Time_Per_Stage = (
                                    Compute["GNR"]["Shared_PRJ"] / Pipeline_Stage
                                ) / (Real_Perf["GNR-Shared_PRJ"] / Pipeline_Stage)
                                Timebreakdown["GNR-Shared_PRJ"] = (
                                    GNR_Shared_PRJ_Time_Per_Stage * Pipeline_Stage
                                )
                                GNR_FFN_PRJ_Time_Per_Stage = (
                                    Compute["GNR"]["FFN_PRJ"] / Pipeline_Stage
                                ) / (Real_Perf["GNR-FFN_PRJ"] / Pipeline_Stage)
                                Timebreakdown["GNR-FFN_PRJ"] = (
                                    GNR_FFN_PRJ_Time_Per_Stage
                                    * Pipeline_Stage
                                    / MOE_Activate
                                    * MM_GP_Average_Max
                                )
                                # GNR_PRJ_Time_Per_Stage = (Compute["GNR"]["PRJ"] / Pipeline_Stage) / (Real_Perf["GNR"]["PRJ"] / Pipeline_Stage)
                                # Timebreakdown["GNR"]["PRJ"] = GNR_PRJ_Time_Per_Stage * Pipeline_Stage / MOE_Activate * MM_GP_Average_Max
                                Timebreakdown["GNR"]["PRJ"] = (
                                    Timebreakdown["GNR-Shared_PRJ"]
                                    + Timebreakdown["GNR-FFN_PRJ"]
                                )
                                GNR_ACT_Time_Per_Stage = (
                                    Compute["GNR"]["ACT"] / Pipeline_Stage
                                ) / (Real_Perf["GNR"]["ACT"] / Pipeline_Stage)
                                Timebreakdown["GNR"]["ACT"] = (
                                    GNR_ACT_Time_Per_Stage * Pipeline_Stage
                                )

                                if (
                                    MM_Card_Per_Stage < 1
                                    or MM_GP_Card_Per_Stage < 1
                                    or MV_Card_Per_Stage < 1
                                ):
                                    raise ValueError(
                                        "Pipeline stage number must be larger than hardware chip number!"
                                    )

                                ## Inter Stage Communication
                                Timebreakdown["OFFCARD"]["PRF"]["inter_Stage"] += (
                                    self.Get_Offcard_Time(
                                        Offcard_Trans["PRF"]["inter_Stage"], 2
                                    )
                                    / 1e6
                                )
                                Timebreakdown["OFFCARD"]["GNR-PRJ"]["inter_Stage"] += (
                                    self.Get_Offcard_Time(
                                        Offcard_Trans["GNR-PRJ"]["inter_Stage"], 2
                                    )
                                    / 1e6
                                )
                                Timebreakdown["OFFCARD"]["GNR-ACT"]["inter_Stage"] += (
                                    self.Get_Offcard_Time(
                                        Offcard_Trans["GNR-ACT"]["inter_Stage"], 2
                                    )
                                    / 1e6
                                )
                                ## Intra Stage Communication (AllReduce)
                                for link in AllReduce_Links:
                                    if link == "DDR":
                                        DDR_Channel = self.hardwares[hard].DDR_Channel
                                        if DDR_Channel == 0:
                                            raise ValueError(
                                                "DDR_Channel must be provided!"
                                            )
                                        size_per_channel = (
                                            Offcard_Trans["PRF"]["intra_Stage"][0]
                                            * MM_Card_Num
                                            / DDR_Channel
                                        )
                                        trans_time = Offcard_Trans["PRF"][
                                            "intra_Stage"
                                        ][1]
                                        Timebreakdown["OFFCARD"]["PRF"][
                                            "intra_Stage"
                                        ] += (
                                            self.Get_Offcard_Time(
                                                [size_per_channel, trans_time], 2, link
                                            )
                                            / 1e6
                                        )
                                        Timebreakdown["OFFCARD"]["PRF"]["AllReduce"][
                                            link
                                        ] = (
                                            self.Get_Offcard_Time(
                                                [size_per_channel, trans_time], 2, link
                                            )
                                            / 1e6
                                        )

                                        size_per_channel = (
                                            Offcard_Trans["GNR-PRJ"]["intra_Stage"][0]
                                            * MM_GP_Card_Num
                                            / DDR_Channel
                                        )
                                        trans_time = Offcard_Trans["GNR-PRJ"][
                                            "intra_Stage"
                                        ][1]
                                        Timebreakdown["OFFCARD"]["GNR-PRJ"][
                                            "intra_Stage"
                                        ] += (
                                            self.Get_Offcard_Time(
                                                [size_per_channel, trans_time], 2, link
                                            )
                                            / 1e6
                                        )
                                        Timebreakdown["OFFCARD"]["GNR-PRJ"][
                                            "AllReduce"
                                        ][link] = (
                                            self.Get_Offcard_Time(
                                                [size_per_channel, trans_time], 2, link
                                            )
                                            / 1e6
                                        )

                                        size_per_channel = (
                                            Offcard_Trans["GNR-ACT"]["intra_Stage"][0]
                                            * MV_Card_Num
                                            / DDR_Channel
                                        )
                                        trans_time = Offcard_Trans["GNR-ACT"][
                                            "intra_Stage"
                                        ][1]
                                        Timebreakdown["OFFCARD"]["GNR-ACT"][
                                            "intra_Stage"
                                        ] += (
                                            self.Get_Offcard_Time(
                                                [size_per_channel, trans_time], 2, link
                                            )
                                            / 1e6
                                        )
                                        Timebreakdown["OFFCARD"]["GNR-ACT"][
                                            "AllReduce"
                                        ][link] = (
                                            self.Get_Offcard_Time(
                                                [size_per_channel, trans_time], 2, link
                                            )
                                            / 1e6
                                        )
                                    else:
                                        Timebreakdown["OFFCARD"]["PRF"][
                                            "intra_Stage"
                                        ] += (
                                            self.Get_Offcard_Time(
                                                Offcard_Trans["PRF"]["intra_Stage"],
                                                MM_Card_Per_Stage,
                                                link,
                                            )
                                            / 1e6
                                        )
                                        Timebreakdown["OFFCARD"]["PRF"]["AllReduce"][
                                            link
                                        ] = (
                                            self.Get_Offcard_Time(
                                                Offcard_Trans["PRF"]["intra_Stage"],
                                                MM_Card_Per_Stage,
                                                link,
                                            )
                                            / 1e6
                                        )
                                        Timebreakdown["OFFCARD"]["GNR-PRJ"][
                                            "intra_Stage"
                                        ] += (
                                            self.Get_Offcard_Time(
                                                Offcard_Trans["GNR-PRJ"]["intra_Stage"],
                                                MM_GP_Card_Per_Stage,
                                                link,
                                            )
                                            / 1e6
                                        )
                                        Timebreakdown["OFFCARD"]["GNR-PRJ"][
                                            "AllReduce"
                                        ][link] = (
                                            self.Get_Offcard_Time(
                                                Offcard_Trans["GNR-PRJ"]["intra_Stage"],
                                                MM_Card_Per_Stage,
                                                link,
                                            )
                                            / 1e6
                                        )
                                        Timebreakdown["OFFCARD"]["GNR-ACT"][
                                            "intra_Stage"
                                        ] += (
                                            self.Get_Offcard_Time(
                                                Offcard_Trans["GNR-ACT"]["intra_Stage"],
                                                MV_Card_Per_Stage,
                                                link,
                                            )
                                            / 1e6
                                        )
                                        Timebreakdown["OFFCARD"]["GNR-ACT"][
                                            "AllReduce"
                                        ][link] = (
                                            self.Get_Offcard_Time(
                                                Offcard_Trans["GNR-ACT"]["intra_Stage"],
                                                MM_Card_Per_Stage,
                                                link,
                                            )
                                            / 1e6
                                        )
                                ## out Card Communication (GPU <-> PIM during attention)
                                Offcard_Trans["PRF"]["out_Card"][0] /= MM_Card_Per_Stage
                                Timebreakdown["OFFCARD"]["PRF"]["out_Card"] += (
                                    self.Get_Offcard_Time(
                                        Offcard_Trans["PRF"]["out_Card"], 2
                                    )
                                    / 1e6
                                )
                                Offcard_Trans["GNR-PRJ"]["out_Card"][
                                    0
                                ] /= MM_GP_Card_Per_Stage
                                Timebreakdown["OFFCARD"]["GNR-PRJ"]["out_Card"] += (
                                    self.Get_Offcard_Time(
                                        Offcard_Trans["GNR-PRJ"]["out_Card"], 2
                                    )
                                    / 1e6
                                )
                                Offcard_Trans["GNR-ACT"]["out_Card"][
                                    0
                                ] /= MV_Card_Per_Stage
                                Timebreakdown["OFFCARD"]["GNR-ACT"]["out_Card"] += (
                                    self.Get_Offcard_Time(
                                        Offcard_Trans["GNR-ACT"]["out_Card"], 2
                                    )
                                    / 1e6
                                )

                                Timebreakdown["OFFCARD"]["PRF"]["Total"] = (
                                    Timebreakdown["OFFCARD"]["PRF"]["inter_Stage"]
                                    + Timebreakdown["OFFCARD"]["PRF"]["intra_Stage"]
                                    + Timebreakdown["OFFCARD"]["PRF"]["out_Card"]
                                )
                                if self.models[model].Parallel_FFN:
                                    Timebreakdown["OFFCARD"]["GNR-PRJ"]["Total"] = (
                                        Timebreakdown["OFFCARD"]["GNR-PRJ"][
                                            "inter_Stage"
                                        ]
                                        + Timebreakdown["OFFCARD"]["GNR-PRJ"][
                                            "intra_Stage"
                                        ]
                                        / 2
                                        + Timebreakdown["OFFCARD"]["GNR-PRJ"][
                                            "out_Card"
                                        ]
                                    )
                                else:
                                    Timebreakdown["OFFCARD"]["GNR-PRJ"]["Total"] = (
                                        Timebreakdown["OFFCARD"]["GNR-PRJ"][
                                            "inter_Stage"
                                        ]
                                        + Timebreakdown["OFFCARD"]["GNR-PRJ"][
                                            "intra_Stage"
                                        ]
                                        + Timebreakdown["OFFCARD"]["GNR-PRJ"][
                                            "out_Card"
                                        ]
                                    )
                                Timebreakdown["OFFCARD"]["GNR-ACT"]["Total"] = (
                                    Timebreakdown["OFFCARD"]["GNR-ACT"]["inter_Stage"]
                                    + Timebreakdown["OFFCARD"]["GNR-ACT"]["intra_Stage"]
                                    + Timebreakdown["OFFCARD"]["GNR-ACT"]["out_Card"]
                                )
                                Timebreakdown["OFFCARD"]["Total"] = (
                                    Timebreakdown["OFFCARD"]["PRF"]["Total"]
                                    + Timebreakdown["OFFCARD"]["GNR-PRJ"]["Total"]
                                    + Timebreakdown["OFFCARD"]["GNR-ACT"]["Total"]
                                )
                                Timebreakdown["Total"] = (
                                    Timebreakdown["PRF"]
                                    + Timebreakdown["GNR"]["PRJ"]
                                    + Timebreakdown["GNR"]["ACT"]
                                    + Timebreakdown["OFFCARD"]["Total"]
                                )

                                Util["PRF"] = (
                                    (Compute["PRF"]["PRJ"] + Compute["PRF"]["ACT"])
                                    * Pipeline_Stage
                                    / (
                                        Timebreakdown["PRF"]
                                        + Timebreakdown["OFFCARD"]["PRF"]["Total"]
                                    )
                                    / MM_TFLOPS
                                )
                                Util["GNR"]["PRJ"] = (
                                    Compute["GNR"]["PRJ"]
                                    * Pipeline_Stage
                                    / (
                                        Timebreakdown["GNR"]["PRJ"]
                                        + Timebreakdown["OFFCARD"]["GNR-PRJ"]["Total"]
                                    )
                                    / MM_GP_TFLOPS
                                )
                                Util["GNR"]["ACT"] = (
                                    Compute["GNR"]["ACT"]
                                    * Pipeline_Stage
                                    / (
                                        Timebreakdown["GNR"]["ACT"]
                                        + Timebreakdown["OFFCARD"]["GNR-ACT"]["Total"]
                                    )
                                    / MV_TFLOPS
                                )
                                if self.hardwares[hard].Form_Type == "Homo":
                                    Util["GNR"]["Total"] = (
                                        (Compute["GNR"]["PRJ"] + Compute["GNR"]["ACT"])
                                        * Pipeline_Stage
                                        / (
                                            Timebreakdown["Total"]
                                            - Timebreakdown["PRF"]
                                            - Timebreakdown["OFFCARD"]["PRF"]["Total"]
                                        )
                                        / MM_GP_TFLOPS
                                    )
                                Util["Total"] = (
                                    Compute["Total"]
                                    * Pipeline_Stage
                                    / Timebreakdown["Total"]
                                    / MM_TFLOPS
                                )

                                maxtoken_info["Compute"] = Compute
                                maxtoken_info["Memory"] = Memory
                                maxtoken_info["Offcard"] = Offcard_Trans
                                maxtoken_info["Arithmetic"] = Arithmetic
                                maxtoken_info["Real_Perf"] = Real_Perf
                                maxtoken_info["Timebreakdown"] = Timebreakdown
                                maxtoken_info["Util"] = Util

                                sparse_info[maxtoken] = maxtoken_info
                            batch_info[sparse] = sparse_info
                        ratio_info[batch_name] = batch_info
                    model_info[ratiop] = ratio_info
                hard_info[model] = model_info
            out[hard] = hard_info
        return out

    def List_Timebreakdown_Ratio(
        self,
        Ratio_P=[0.25],
        Batchsize=[32],
        Max_Token=[],
        Model=[],
        Hardware=[],
        Sparse=[],
        Pipeline_Stage=1,
        first_col=True,
        SLO_scale=5,
        request_rate=1.5,
    ):
        if Model == []:
            Model = [i for i in self.models.keys()]
        if Hardware == []:
            Hardware = [i for i in self.hardwares.keys()]
        if Sparse == []:
            Sparse = [i for i in self.sparses.keys()]

        all_data = self.Compute_Timebreakdown(
            Ratio_P, Batchsize, Max_Token, Model, Hardware, Sparse, Pipeline_Stage
        )
        if first_col:
            print("\033[1;31m*******List_Timebreakdown*******\033[0m")
            # print("\033[1;31m%-8s%-20s%-12s%-8s%-7s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-12s%-13s%-10s%-10s%-10s%-15s\033[0m"% \
            # print("\033[1;31m%-8s%-20s%-5s%-8s%-7s%-7s%-2s%-15s%-15s%-15s%-12s%-2s%-12s%-2s%-12s%-12s%-2s%-12s%-12s%-12s%-2s%-12s%-12s%-12s%-12s%-12s%-12s%-2s%-12s%-12s%-12s%-12s%-2s\033[0m"% \
            print(
                "\033[1;31m%-8s%-20s%-5s%-8s%-10s%-7s%-2s%-15s%-15s%-15s%-12s%-12s%-2s%-15s%-2s%-12s%-12s%-2s%-12s%-12s%-12s%-2s%-12s%-12s%-12s%-12s%-12s%-12s%-2s%-12s%-12s%-12s%-12s\033[0m"
                % (
                    "Model",
                    "Hardware",
                    "Pipe",
                    "Ratio_P",
                    "Batch",
                    "Seq",
                    "|",  # "Comp-P", "Arith-P", "Time-P", "Util-P", "Comp-GP", "Arith-P", "Time-GP", "Util-GP", "Comp-GA", "Arith-P", "Time-GA", "Util-GA", "Comp-Ttl", "Time-Offcard", "Time-Ttl", "Util-Ttl", "Token/Sec", "Token/Sec/Batch" ))
                    # "Comp-P", "Arith-P", "Time-P", "Util-P", "Comp-GP", "Arith-P", "Time-GP", "Util-GP", "Comp-GA", "Arith-P", "Time-GA", "Util-GA", "Comp-Ttl", "Time-Comm", "Time-Ttl", "Util-Ttl", "OutToken/Sec", "AllToken/Sec"))
                    # "Comp-P", "CompTime-P", "CommTime-P", "Time-P", "Comp-GP", "CompTime-GP", "CommTime-GP", "Time-GP", "Comp-GA", "CompTime-GA", "CommTime-GA", "Time-GA", "CompTime-Ttl", "CommTime-Ttl", "Time-Ttl", "Util-Ttl", "OutToken/Sec", "AllToken/Sec"))
                    "FPRT(ms)",
                    "L(ms/tok)",
                    "Thr(quary/s)",
                    "SLO",
                    "\xa2/k-token",
                    "|",
                    "BatchTime-Ttl",
                    "|",
                    "Comp(%)",
                    "Comm(%)",
                    "|",
                    "P(%)",
                    "GP(%)",
                    "GA(%)",
                    "|",
                    "Comp-P(%)",
                    "Comm-P(%)",
                    "Comp-GP(%)",
                    "Comm-GP(%)",
                    "Comp-GA(%)",
                    "Comm-GA(%)",
                    "|",
                    "Util-Ttl",
                    "Util-P",
                    "Util-GP",
                    "Util-GA",
                )
            )

        for hard in Hardware:
            if hard == None:
                continue
            hard_info = all_data[hard]
            for model in Model:
                model_info = hard_info[model]
                for ratiop in Ratio_P:
                    ratio_info = model_info[ratiop]
                    for batch in Batchsize:
                        batch_info = ratio_info[batch]
                        batch_name = str(batch)
                        for sparse in Sparse:
                            sparse_info = batch_info[sparse]
                            if Max_Token == []:
                                Max_Token_List = [self.models[model].Max_Token]
                            else:
                                Max_Token_List = Max_Token
                            for maxtoken in Max_Token_List:
                                maxtoken_info = sparse_info[maxtoken]
                                Parallel_str = str(Pipeline_Stage)
                                Throughput_Ratio = Pipeline_Stage
                                self.cost_model.init_hardware(self.hardwares[hard])
                                # print("%-8s%-20s%-12s%-8.2f%-7d%-10d%-10s%-10.2f%-11.2f%-11.2f%-10.4f%-10.2f%-12.2f%-12.2f%-10.4f%-10.2f%-12.2f%-12.2f%-10.4f%-13.2f%-13.2f%-10.2f%-10s%-15.2f%-15.2f" % \
                                #     (model, hard, Parallel_str,ratiop, batch, maxtoken, sparse,
                                #     maxtoken_info["Compute"]["PRF"]["PRJ"] + maxtoken_info["Compute"]["PRF"]["ACT"], maxtoken_info["Arithmetic"]["PRF"], maxtoken_info["Timebreakdown"]["PRF"], maxtoken_info["Util"]["PRF"], # CompTime-P
                                #     maxtoken_info["Compute"]["GNR"]["PRJ"], maxtoken_info["Arithmetic"]["GNR"]["PRJ"], maxtoken_info["Timebreakdown"]["GNR"]["PRJ"], maxtoken_info["Util"]["GNR"]["PRJ"],
                                #     maxtoken_info["Compute"]["GNR"]["ACT"], maxtoken_info["Arithmetic"]["GNR"]["ACT"], maxtoken_info["Timebreakdown"]["GNR"]["ACT"], maxtoken_info["Util"]["GNR"]["ACT"],
                                #     maxtoken_info["Compute"]["Total"], maxtoken_info["Timebreakdown"]["OFFCARD"], maxtoken_info["Timebreakdown"]["Total"], str(int(maxtoken_info["Util"]["Total"]*1000)/1000) if self.hardwares[hard].Form_Type=="Homo" else "NA",
                                #     # batch * maxtoken * (1-ratiop) / (maxtoken_info["Timebreakdown"]["Total"]), batch * maxtoken * (1-ratiop) / (maxtoken_info["Timebreakdown"]["Total"]) / batch))
                                #     batch * maxtoken * (1-ratiop) * Throughput_Ratio / (maxtoken_info["Timebreakdown"]["Total"]),
                                #     batch * maxtoken * Throughput_Ratio / (maxtoken_info["Timebreakdown"]["Total"])))
                                # print("%-8s%-20s%-5s%-8.2f%-7d%-7d%-10.2f%-11.2f%-11.2f%-10.4f%-10.2f%-12.2f%-12.2f%-10.4f%-10.2f%-12.2f%-12.2f%-10.4f%-13.2f%-13.2f%-10.2f%-10s%-15.2f%-15.2f" % \
                                # print("\033[1;31m%-8s%-20s%-5s%-8.2f%-7d%-7d%-2s%-15.2f%-15.2f%-15.2f%-12.2f%-2s%-12.2f%-2s%-12.2f%-12.2f%%%-2s%-12.2f%%%-12.2f%%%-12.2f%%%-2s%-12.2f%%%-12.2f%%%-12.2f%%%-12.2f%%%-12.2f%%%-12.2f%%%-2s%-12.2f%%%-12.2f%%%-12.2f%%%-12.2f%%\033[0m"% \
                                if batch_name in ["Max", "max", "MAX"]:
                                    batch = maxtoken_info["batchsize"]
                                    batch_name_print = "max({})".format(batch)
                                else:
                                    batch_name_print = "{}".format(batch)
                                print(
                                    "{:<8}{:<20}{:<5}{:<8.2f}{:<10}{:<7d}{:<2}{:<15.2f}{:<15.2f}{:<15.2f}{:<12.5%}\xa2{:<11.6}{:<2}{:>13.2f}  {:<2}{:<12.2%}{:<12.2%}{:2}{:<12.2%}{:<12.2%}{:<12.2%}{:2}{:<12.2%}{:<12.2%}{:<12.2%}{:<12.2%}{:<12.2%}{:<12.2%}{:2}{:<12.2%}{:<12.2%}{:<12.2%}{:<12.2%}".format(
                                        model,
                                        hard,
                                        Parallel_str,
                                        ratiop,
                                        batch_name_print,
                                        maxtoken,
                                        "|",
                                        (
                                            maxtoken_info["Timebreakdown"]["PRF"]
                                            + maxtoken_info["Timebreakdown"]["OFFCARD"][
                                                "PRF"
                                            ]["Total"]
                                        )
                                        * 1000,  # Time-P
                                        Throughput_Ratio
                                        * (
                                            maxtoken_info["Timebreakdown"]["Total"]
                                            - maxtoken_info["Timebreakdown"]["PRF"]
                                            - maxtoken_info["Timebreakdown"]["OFFCARD"][
                                                "PRF"
                                            ]["Total"]
                                        )
                                        / (maxtoken * (1 - ratiop))
                                        * 1000,  # ms for per output token of a quary
                                        batch
                                        * Throughput_Ratio
                                        / (
                                            maxtoken_info["Timebreakdown"]["Total"]
                                        ),  # quary/s
                                        self.Calc_SLO(
                                            request_rate,
                                            maxtoken_info["Timebreakdown"]["Total"]
                                            * SLO_scale,
                                            maxtoken_info["Timebreakdown"]["Total"],
                                            maxtoken_info["Timebreakdown"]["PRF"]
                                            + maxtoken_info["Timebreakdown"]["OFFCARD"][
                                                "PRF"
                                            ]["Total"],
                                            batch,
                                        ),  # SLO
                                        100
                                        * 1000
                                        * self.cost_model.cal_total_cost(
                                            maxtoken
                                            * batch
                                            * Throughput_Ratio
                                            / (maxtoken_info["Timebreakdown"]["Total"])
                                        ),
                                        "|",
                                        maxtoken_info["Timebreakdown"]["Total"],
                                        "|",  # Time-Ttl
                                        (
                                            maxtoken_info["Timebreakdown"]["PRF"]
                                            + maxtoken_info["Timebreakdown"]["GNR"][
                                                "PRJ"
                                            ]
                                            + maxtoken_info["Timebreakdown"]["GNR"][
                                                "ACT"
                                            ]
                                        )
                                        / maxtoken_info["Timebreakdown"][
                                            "Total"
                                        ],  # CompTime-Ttl
                                        maxtoken_info["Timebreakdown"]["OFFCARD"][
                                            "Total"
                                        ]
                                        / maxtoken_info["Timebreakdown"]["Total"],
                                        "|",  # CommTime-Ttl
                                        (
                                            maxtoken_info["Timebreakdown"]["PRF"]
                                            + maxtoken_info["Timebreakdown"]["OFFCARD"][
                                                "PRF"
                                            ]["Total"]
                                        )
                                        / maxtoken_info["Timebreakdown"][
                                            "Total"
                                        ],  # Time-P
                                        (
                                            maxtoken_info["Timebreakdown"]["GNR"]["PRJ"]
                                            + maxtoken_info["Timebreakdown"]["OFFCARD"][
                                                "GNR-PRJ"
                                            ]["Total"]
                                        )
                                        / maxtoken_info["Timebreakdown"][
                                            "Total"
                                        ],  # Time-GP
                                        (
                                            maxtoken_info["Timebreakdown"]["GNR"]["ACT"]
                                            + maxtoken_info["Timebreakdown"]["OFFCARD"][
                                                "GNR-ACT"
                                            ]["Total"]
                                        )
                                        / maxtoken_info["Timebreakdown"][
                                            "Total"
                                        ],  # Time-GA
                                        "|",
                                        maxtoken_info["Timebreakdown"]["PRF"]
                                        / maxtoken_info["Timebreakdown"][
                                            "Total"
                                        ],  # CompTime-P
                                        maxtoken_info["Timebreakdown"]["OFFCARD"][
                                            "PRF"
                                        ]["Total"]
                                        / maxtoken_info["Timebreakdown"][
                                            "Total"
                                        ],  # CommTime-P
                                        maxtoken_info["Timebreakdown"]["GNR"]["PRJ"]
                                        / maxtoken_info["Timebreakdown"][
                                            "Total"
                                        ],  # CompTime-GP
                                        maxtoken_info["Timebreakdown"]["OFFCARD"][
                                            "GNR-PRJ"
                                        ]["Total"]
                                        / maxtoken_info["Timebreakdown"][
                                            "Total"
                                        ],  # CommTime-GP
                                        maxtoken_info["Timebreakdown"]["GNR"]["ACT"]
                                        / maxtoken_info["Timebreakdown"][
                                            "Total"
                                        ],  # CompTime-GA
                                        maxtoken_info["Timebreakdown"]["OFFCARD"][
                                            "GNR-ACT"
                                        ]["Total"]
                                        / maxtoken_info["Timebreakdown"][
                                            "Total"
                                        ],  # CommTime-GA
                                        "|",
                                        maxtoken_info["Util"]["Total"],  # Util-Ttl
                                        maxtoken_info["Util"]["PRF"],
                                        maxtoken_info["Util"]["GNR"]["PRJ"],
                                        maxtoken_info["Util"]["GNR"]["ACT"],
                                    )
                                )
                                if (
                                    maxtoken_info["Offcard"]["GNR-PRJ"]["intra_Stage"][
                                        0
                                    ]
                                    / 1e6
                                    > 2
                                ):
                                    print(
                                        "Comm BW bound: {:.2f}MB".format(
                                            maxtoken_info["Offcard"]["GNR-PRJ"][
                                                "intra_Stage"
                                            ][0]
                                            / 1e6
                                        )
                                    )
                                # print("Token/Sec: {}".format(maxtoken * batch * Throughput_Ratio / (maxtoken_info["Timebreakdown"]["Total"])))
                                """
                                    maxtoken_info["Compute"]["PRF"]["PRJ"] + maxtoken_info["Compute"]["PRF"]["ACT"], # Comp-P
                                    maxtoken_info["Timebreakdown"]["PRF"], # CompTime-P
                                    maxtoken_info["Timebreakdown"]["OFFCARD"]["PRF"]["Total"], # CommTime-P
                                    maxtoken_info["Timebreakdown"]["PRF"] + maxtoken_info["Timebreakdown"]["OFFCARD"]["PRF"]["Total"], # Time-P

                                    maxtoken_info["Compute"]["GNR"]["PRJ"], # Comp-GP
                                    maxtoken_info["Timebreakdown"]["GNR"]["PRJ"], # CompTime-GP
                                    maxtoken_info["Timebreakdown"]["OFFCARD"]["GNR-PRJ"]["Total"], # CommTime-GP
                                    maxtoken_info["Timebreakdown"]["GNR"]["PRJ"] + maxtoken_info["Timebreakdown"]["OFFCARD"]["GNR-PRJ"]["Total"], # Time-GP

                                    maxtoken_info["Compute"]["GNR"]["ACT"], # Comp-GA
                                    maxtoken_info["Timebreakdown"]["GNR"]["ACT"], # CompTime-GA
                                    maxtoken_info["Timebreakdown"]["OFFCARD"]["GNR-ACT"]["Total"], # CommTime-GA
                                    maxtoken_info["Timebreakdown"]["GNR"]["ACT"] + maxtoken_info["Timebreakdown"]["OFFCARD"]["GNR-ACT"]["Total"], # Time-GA

                                    maxtoken_info["Timebreakdown"]["PRF"] + maxtoken_info["Timebreakdown"]["GNR"]["PRJ"] + maxtoken_info["Timebreakdown"]["GNR"]["ACT"],# CompTime-Ttl
                                    maxtoken_info["Timebreakdown"]["OFFCARD"]["Total"],# CommTime-Ttl
                                    maxtoken_info["Timebreakdown"]["Total"],# Time-Ttl

                                    str(int(maxtoken_info["Util"]["Total"]*1000)/1000) if self.hardwares[hard].Form_Type=="Homo" else "NA",# Util-Ttl
                                    batch * maxtoken * (1-ratiop) * Throughput_Ratio / (maxtoken_info["Timebreakdown"]["Total"]),# OutToken/Sec
                                    batch * maxtoken * Throughput_Ratio / (maxtoken_info["Timebreakdown"]["Total"]))) # AllToken/Sec
                                """

    def List_Timebreakdown_Absolute(
        self,
        Ratio_P=[0.25],
        Batchsize=[32],
        Max_Token=[],
        Model=[],
        Hardware=[],
        Sparse=[],
        Pipeline_Stage=1,
        first_col=True,
        SLO_scale=5,
        request_rate=1.5,
    ):
        if Model == []:
            Model = [i for i in self.models.keys()]
        if Hardware == []:
            Hardware = [i for i in self.hardwares.keys()]
        if Sparse == []:
            Sparse = [i for i in self.sparses.keys()]

        all_data = self.Compute_Timebreakdown(
            Ratio_P, Batchsize, Max_Token, Model, Hardware, Sparse, Pipeline_Stage
        )
        if first_col:
            print("\033[1;31m*******List_Timebreakdown*******\033[0m")
            print(
                "\033[1;31m%-15s%-20s%-12s%-8s%-10s%-10s%-10s%-10s%-11s%-11s%-10s%-10s%-12s%-12s%-10s%-10s%-12s%-12s%-10s%-13s%-13s%-10s%-10s%-15s%-15s\033[0m"
                % (
                    "Model",
                    "Hardware",
                    "Pipe_Stage",
                    "Ratio_P",
                    "Batch",
                    "Max_Token",
                    "Sparse",
                    "Comp-P",
                    "CompTime-P",
                    "CommTime-P",
                    "Time-P",
                    "Comp-GP",
                    "CompTime-GP",
                    "CommTime-GP",
                    "Time-GP",
                    "Comp-GA",
                    "CompTime-GA",
                    "CommTime-GA",
                    "Time-GA",
                    "CompTime-Ttl",
                    "CommTime-Ttl",
                    "Time-Ttl",
                    "Util-Ttl",
                    "OutToken/Sec",
                    "AllToken/Sec",
                )
            )

        for hard in Hardware:
            if hard == None:
                continue
            hard_info = all_data[hard]
            for model in Model:
                model_info = hard_info[model]
                for ratiop in Ratio_P:
                    ratio_info = model_info[ratiop]
                    for batch in Batchsize:
                        batch_name = str(batch)
                        batch_info = ratio_info[batch]
                        for sparse in Sparse:
                            sparse_info = batch_info[sparse]
                            if Max_Token == []:
                                Max_Token_List = [self.models[model].Max_Token]
                            else:
                                Max_Token_List = Max_Token
                            for maxtoken in Max_Token_List:
                                maxtoken_info = sparse_info[maxtoken]
                                Parallel_str = str(Pipeline_Stage)
                                Throughput_Ratio = Pipeline_Stage
                                self.cost_model.init_hardware(self.hardwares[hard])
                                if batch_name in ["Max", "max", "MAX"]:
                                    batch = maxtoken_info["batchsize"]
                                    batch_name_print = "max({})".format(batch)
                                else:
                                    batch_name_print = "{}".format(batch)

                                print(
                                    "%-15s%-20s%-12s%-8.2f%-10s%-10d%-10s%-10.2f%-11.2f%-11.2f%-10.2f%-10.2f%-12.2f%-12.2f%-10.2f%-10.2f%-12.2f%-12.2f%-10.2f%-13.2f%-13.2f%-10.2f%-10s%-15.2f%-15.2f"
                                    % (
                                        model,
                                        hard,
                                        Parallel_str,
                                        ratiop,
                                        batch_name_print,
                                        maxtoken,
                                        sparse,
                                        maxtoken_info["Compute"]["PRF"]["PRJ"]
                                        + maxtoken_info["Compute"]["PRF"][
                                            "ACT"
                                        ],  # Comp-P
                                        maxtoken_info["Timebreakdown"][
                                            "PRF"
                                        ],  # CompTime-P
                                        maxtoken_info["Timebreakdown"]["OFFCARD"][
                                            "PRF"
                                        ][
                                            "Total"
                                        ],  # CommTime-P
                                        maxtoken_info["Timebreakdown"]["PRF"]
                                        + maxtoken_info["Timebreakdown"]["OFFCARD"][
                                            "PRF"
                                        ][
                                            "Total"
                                        ],  # Time-P
                                        maxtoken_info["Compute"]["GNR"][
                                            "PRJ"
                                        ],  # Comp-GP
                                        maxtoken_info["Timebreakdown"]["GNR"][
                                            "PRJ"
                                        ],  # CompTime-GP
                                        maxtoken_info["Timebreakdown"]["OFFCARD"][
                                            "GNR-PRJ"
                                        ][
                                            "Total"
                                        ],  # CommTime-GP
                                        maxtoken_info["Timebreakdown"]["GNR"]["PRJ"]
                                        + maxtoken_info["Timebreakdown"]["OFFCARD"][
                                            "GNR-PRJ"
                                        ][
                                            "Total"
                                        ],  # Time-GP
                                        maxtoken_info["Compute"]["GNR"][
                                            "ACT"
                                        ],  # Comp-GA
                                        maxtoken_info["Timebreakdown"]["GNR"][
                                            "ACT"
                                        ],  # CompTime-GA
                                        maxtoken_info["Timebreakdown"]["OFFCARD"][
                                            "GNR-ACT"
                                        ][
                                            "Total"
                                        ],  # CommTime-GA
                                        maxtoken_info["Timebreakdown"]["GNR"]["ACT"]
                                        + maxtoken_info["Timebreakdown"]["OFFCARD"][
                                            "GNR-ACT"
                                        ][
                                            "Total"
                                        ],  # Time-GA
                                        maxtoken_info["Timebreakdown"]["PRF"]
                                        + maxtoken_info["Timebreakdown"]["GNR"]["PRJ"]
                                        + maxtoken_info["Timebreakdown"]["GNR"][
                                            "ACT"
                                        ],  # CompTime-Ttl
                                        maxtoken_info["Timebreakdown"]["OFFCARD"][
                                            "Total"
                                        ],  # CommTime-Ttl
                                        maxtoken_info["Timebreakdown"][
                                            "Total"
                                        ],  # Time-Ttl
                                        (
                                            str(
                                                int(
                                                    maxtoken_info["Util"]["Total"]
                                                    * 1000
                                                )
                                                / 1000
                                            )
                                            if self.hardwares[hard].Form_Type == "Homo"
                                            else "NA"
                                        ),  # Util-Ttl
                                        batch
                                        * maxtoken
                                        * (1 - ratiop)
                                        * Throughput_Ratio
                                        / (
                                            maxtoken_info["Timebreakdown"]["Total"]
                                        ),  # OutToken/Sec
                                        batch
                                        * maxtoken
                                        * Throughput_Ratio
                                        / (maxtoken_info["Timebreakdown"]["Total"]),
                                    )
                                )  # AllToken/Sec
                                # print(maxtoken_info["Real_Perf"]["GNR"]["PRJ"], maxtoken_info["Real_Perf"]["GNR"]["ACT"])
                                # print("Prefill Comm:", maxtoken_info["Timebreakdown"]["OFFCARD"]["PRF"]["AllReduce"])
                                # print("Generation Comm:", maxtoken_info["Timebreakdown"]["OFFCARD"]["GNR-PRJ"]["AllReduce"])

    def Draw_Homo_Roofline(
        self,
        Ratio_P=[0.25],
        Batchsize=[32],
        Max_Token=[],
        Model=[],
        Hardware=[],
        Sparse=[],
    ):
        if Model == []:
            Model = [self.models.keys()[0]]
        if Hardware == []:
            Hardware = [
                i if self.hardwares[i].Form_Type == "Homo" else None
                for i in self.hardwares.keys()
            ]
        if Sparse == []:
            Sparse = [i for i in self.sparses.keys()]

        self.List_Timebreakdown(Ratio_P, Batchsize, Max_Token, Model, Hardware, Sparse)

        all_data = self.Compute_Timebreakdown(
            Ratio_P, Batchsize, Max_Token, Model, Hardware, Sparse
        )
        fig = plt.figure(dpi=300)
        for hard in Hardware:
            if hard == None:
                continue
            MM_TFLOPS = self.hardwares[hard].MM_TFLOPS
            MM_BW_TBs = self.hardwares[hard].MM_BW_TBs
            MM_Sweet_Point = self.hardwares[hard].MM_Sweet_Point
            x = [MM_Sweet_Point / 10000, MM_Sweet_Point, 10000]
            y = [MM_TFLOPS / 10000, MM_TFLOPS, MM_TFLOPS]
            plt.loglog(x, y, lw=1, ls="-", label=hard)
        title = ""
        for hard in Hardware:
            if hard == None:
                continue
            hard_info = all_data[hard]
            for model in Model:
                model_info = hard_info[model]
                model_legend = ""
                model_title = ""
                if len(Model) != 1:
                    model_legend = model + " "
                else:
                    model_title = model + " "
                for ratiop in Ratio_P:
                    ratio_info = model_info[ratiop]
                    ratiop_legend = ""
                    ratiop_title = ""
                    if len(Ratio_P) != 1:
                        ratiop_legend = "Ratio_P=" + str(ratiop) + " "
                    else:
                        ratiop_title = "Ratio_P=" + str(ratiop) + " "
                    for batch in Batchsize:
                        batch_info = ratio_info[batch]
                        batchsize_legend = ""
                        batchsize_title = ""
                        # if len(Batchsize) != 1: batchsize_legend = "Batch=" + str(batch) + " "
                        # else: batchsize_title = "Batch=" + str(batch) + " "
                        for sparse in Sparse:
                            sparse_info = batch_info[sparse]
                            sparse_legend = ""
                            sparse_title = ""
                            if len(Sparse) != 1:
                                sparse_legend = "Sparse=" + str(sparse) + " "
                            else:
                                sparse_title = "Sparse=" + str(sparse) + " "
                            if Max_Token == []:
                                Max_Token_List = [self.models[model].Max_Token]
                            else:
                                Max_Token_List = Max_Token
                            for maxtoken in Max_Token_List:
                                maxtoken_legend = ""
                                maxtoken_title = ""
                                if len(Max_Token_List) != 1:
                                    maxtoken_legend = "Length=" + str(maxtoken) + " "
                                else:
                                    maxtoken_title = "Length=" + str(maxtoken) + " "
                                maxtoken_info = sparse_info[maxtoken]

                                if batch in ["Max", "max", "MAX"]:
                                    batchsize = "max({})".format(
                                        maxtoken_info["batchsize"]
                                    )
                                else:
                                    batchsize = str(batch)
                                if len(Batchsize) != 1 or batch in [
                                    "max",
                                    "Max",
                                    "MAX",
                                ]:
                                    batchsize_legend = "B=" + batchsize + " "
                                else:
                                    batchsize_title = "B=" + batchsize + " "

                                x = [
                                    maxtoken_info["Arithmetic"]["Total"],
                                    maxtoken_info["Arithmetic"]["Total"],
                                ]
                                y = [1, 2000]
                                plt.loglog(
                                    x,
                                    y,
                                    lw=1,
                                    ls="--",
                                    label=model_legend
                                    + ratiop_legend
                                    + batchsize_legend
                                    + sparse_legend
                                    + maxtoken_legend,
                                )
                                title = (
                                    model_title
                                    + ratiop_title
                                    + batchsize_title
                                    + sparse_title
                                    + maxtoken_title
                                )
            break

        fontsize = 8
        plt.xlim(1, 10000)
        plt.ylim(1, 2000)
        plt.tick_params(labelsize=fontsize)
        plt.xlabel("Arithmetic Intensity(OP/Byte)", fontsize=fontsize)
        plt.ylabel("Performance(TFLOPS)", fontsize=fontsize)
        plt.title("Roofline " + title, fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.show()

    def Draw_Homo_Util(
        self,
        Ratio_P=[0.25],
        Batchsize=[32],
        Max_Token=[],
        Model=[],
        Hardware=[],
        Sparse=[],
    ):
        if Model == []:
            Model = [i for i in self.models.keys()]
        if Hardware == []:
            for hardname, hard in self.hardwares.items():
                if hard.Form_Type == "Homo":
                    Hardware.append(hardname)
        if Sparse == []:
            Sparse = [i for i in self.sparses.keys()]

        self.List_Timebreakdown(Ratio_P, Batchsize, Max_Token, Model, Hardware, Sparse)

        all_data = self.Compute_Timebreakdown(
            Ratio_P, Batchsize, Max_Token, Model, Hardware, Sparse
        )
        fig = plt.figure(dpi=300)
        title = ""
        hard_index = model_index = ratiop_index = batch_index = maxtoken_index = (
            sparse_index
        ) = 0
        max_util = 0
        for hard in Hardware:
            if hard == None:
                continue
            x = []
            y = []
            hard_info = all_data[hard]
            hard_legend = ""
            hard_title = ""
            if len(Hardware) != 1:
                hard_legend = hard + " "
            else:
                hard_title = hard + ""
            model_index = 0
            for model in Model:
                model_info = hard_info[model]
                model_legend = ""
                model_title = ""
                if len(Model) != 1:
                    model_legend = model + " "
                else:
                    model_title = model + ""
                ratiop_index = 0
                for ratiop in Ratio_P:
                    ratio_info = model_info[ratiop]
                    ratiop_legend = ""
                    ratiop_title = ""
                    if len(Ratio_P) != 1:
                        ratiop_legend = "Ratio_P=" + str(ratiop) + " "
                    else:
                        ratiop_title = "Ratio_P=" + str(ratiop) + " "
                    batch_index = 0
                    for batch in Batchsize:
                        batch_info = ratio_info[batch]
                        batchsize_legend = ""
                        batchsize_title = ""
                        # if len(Batchsize) != 1: batchsize_legend = "Batch=" + str(batch) + " "
                        # else: batchsize_title = "Batch=" + str(batch) + " "
                        sparse_index = 0
                        for sparse in Sparse:
                            sparse_info = batch_info[sparse]
                            sparse_legend = ""
                            sparse_title = ""
                            if len(Sparse) != 1:
                                sparse_legend = "Sparse=" + str(sparse) + " "
                            else:
                                sparse_title = "Sparse=" + str(sparse) + " "
                            if Max_Token == []:
                                Max_Token_List = [self.models[model].Max_Token]
                            else:
                                Max_Token_List = Max_Token
                            maxtoken_index = 0
                            for maxtoken in Max_Token_List:
                                maxtoken_legend = ""
                                maxtoken_title = ""
                                if len(Max_Token_List) != 1:
                                    maxtoken_legend = "Length=" + str(maxtoken) + " "
                                elif Max_Token != []:
                                    maxtoken_title = "Length=" + str(maxtoken) + " "

                                maxtoken_info = sparse_info[maxtoken]

                                if batch in ["Max", "max", "MAX"]:
                                    batchsize = "max({})".format(
                                        maxtoken_info["batchsize"]
                                    )
                                else:
                                    batchsize = str(batch)
                                if len(Batchsize) != 1 or batch in [
                                    "max",
                                    "Max",
                                    "MAX",
                                ]:
                                    batchsize_legend = "B=" + batchsize + " "
                                else:
                                    batchsize_title = "B=" + batchsize + " "

                                x.append(
                                    len(Hardware)
                                    * len(Ratio_P)
                                    * len(Batchsize)
                                    * len(Sparse)
                                    * len(Max_Token_List)
                                    * model_index
                                    + model_index
                                    + len(Ratio_P)
                                    * len(Batchsize)
                                    * len(Sparse)
                                    * len(Max_Token_List)
                                    * hard_index
                                    + len(Batchsize)
                                    * len(Sparse)
                                    * len(Max_Token_List)
                                    * ratiop_index
                                    + len(Sparse) * len(Max_Token_List) * batch_index
                                    + len(Max_Token_List) * sparse_index
                                    + maxtoken_index
                                )
                                y.append(maxtoken_info["Util"]["Total"])
                                title = (
                                    hard_title
                                    + model_title
                                    + ratiop_title
                                    + batchsize_title
                                    + sparse_title
                                    + maxtoken_title
                                )
                                maxtoken_index += 1
                            sparse_index += 1
                        batch_index += 1
                    ratiop_index += 1

                model_index += 1

            plt.bar(x, y, width=0.9, align="center", label=hard)
            max_util = max_util if max_util > max(y) else max(y)
            # print (x)
            # print (y)
            hard_index += 1

        fontsize = 8
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)

        for model_index in range(len(Model)):
            plt.text(
                len(Hardware)
                * len(Ratio_P)
                * len(Batchsize)
                * len(Sparse)
                * len(Max_Token_List)
                * model_index
                + model_index,
                0 - max_util * 0.04,
                Model[model_index],
                fontsize=fontsize,
            )

        plt.tick_params(labelsize=fontsize)
        # plt.xlabel("", fontsize=fontsize)

        plt.ylabel("Utilization", fontsize=fontsize)
        plt.title("Utilization " + title, fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.show()

    def Draw_Timebreakdown(
        self,
        Ratio_P=[0.25],
        Batchsize=[32],
        Max_Token=[],
        Model=[],
        Hardware=[],
        Sparse=[],
        Parallel="Model",
    ):
        if Model == []:
            Model = [i for i in self.models.keys()]
        if Hardware == []:
            Hardware = [i for i in self.hardwares.keys()]
        if Sparse == []:
            Sparse = [i for i in self.sparses.keys()]

        self.List_Timebreakdown(Ratio_P, Batchsize, Max_Token, Model, Hardware, Sparse)
        all_data = self.Compute_Timebreakdown(
            Ratio_P, Batchsize, Max_Token, Model, Hardware, Sparse
        )

        fig = plt.figure(dpi=300, figsize=[12, 4])
        title = ""
        PRF_Time = []
        GNR_PRJ_Time = []
        GNR_ACT_Time = []
        OFFCARD_Time = []
        Legends = []

        for hard in Hardware:
            if hard == None:
                continue
            hard_info = all_data[hard]
            hard_legend = ""
            hard_title = ""
            if len(Hardware) != 1:
                hard_legend = hard + " "
            else:
                hard_title = hard + " "
            model_index = 0
            for model in Model:
                model_info = hard_info[model]
                model_legend = ""
                model_title = ""
                if len(Model) != 1:
                    model_legend = model + " "
                else:
                    model_title = model + " "
                ratiop_index = 0
                for ratiop in Ratio_P:
                    ratio_info = model_info[ratiop]
                    ratiop_legend = ""
                    ratiop_title = ""
                    if len(Ratio_P) != 1:
                        ratiop_legend = "RP=" + str(ratiop) + " "
                    else:
                        ratiop_title = "RP=" + str(ratiop) + " "
                    batch_index = 0
                    for batch in Batchsize:
                        batch_info = ratio_info[batch]
                        batchsize_legend = ""
                        batchsize_title = ""
                        # if len(Batchsize) != 1: batchsize_legend = "B=" + str(batch) + " "
                        # else: batchsize_title = "B=" + str(batch) + " "
                        for sparse in Sparse:
                            sparse_info = batch_info[sparse]
                            sparse_legend = ""
                            sparse_title = ""
                            # if len(Sparse) != 1: sparse_legend = "S=" + str(sparse) + " "
                            if len(Sparse) != 1:
                                sparse_legend += str(sparse) + " "
                            else:
                                sparse_title = "S=" + str(sparse) + " "
                            if Max_Token == []:
                                Max_Token_List = [self.models[model].Max_Token]
                            else:
                                Max_Token_List = Max_Token
                            maxtoken_index = 0
                            for maxtoken in Max_Token_List:
                                maxtoken_legend = ""
                                maxtoken_title = ""
                                if len(Max_Token_List) != 1:
                                    maxtoken_legend = "L=" + str(maxtoken) + " "
                                elif Max_Token != []:
                                    maxtoken_title = "L=" + str(maxtoken) + " "

                                maxtoken_info = sparse_info[maxtoken]

                                if batch in ["Max", "max", "MAX"]:
                                    batchsize = "max({})".format(
                                        maxtoken_info["batchsize"]
                                    )
                                else:
                                    batchsize = str(batch)
                                if len(Batchsize) != 1 or batch in [
                                    "max",
                                    "Max",
                                    "MAX",
                                ]:
                                    batchsize_legend = "B=" + batchsize + " "
                                else:
                                    batchsize_title = "B=" + batchsize + " "

                                PRF_Time.append(maxtoken_info["Timebreakdown"]["PRF"])
                                GNR_PRJ_Time.append(
                                    maxtoken_info["Timebreakdown"]["GNR"]["PRJ"]
                                )
                                GNR_ACT_Time.append(
                                    maxtoken_info["Timebreakdown"]["GNR"]["ACT"]
                                )
                                OFFCARD_Time.append(
                                    maxtoken_info["Timebreakdown"]["OFFCARD"]["Total"]
                                )
                                title = (
                                    hard_title
                                    + model_title
                                    + ratiop_title
                                    + batchsize_title
                                    + sparse_title
                                    + maxtoken_title
                                )
                                Legends.append(
                                    model_legend
                                    + ratiop_legend
                                    + batchsize_legend
                                    + sparse_legend
                                    + maxtoken_legend
                                )
            PRF_Time.append(0)
            GNR_PRJ_Time.append(0)
            GNR_ACT_Time.append(0)
            OFFCARD_Time.append(0)
            Legends.append("")

        fontsize = 75 / len(Legends)
        plt.grid(axis="x", zorder=0, linestyle="--")

        ax = plt.gca()
        ax.axes.yaxis.set_visible(False)
        index = np.arange(0, len(PRF_Time))
        # plt.barh(index, PRF_Time, height = 0.9, align = "edge", label = "Prefill", zorder=100, log="True")
        # plt.barh(index, GNR_PRJ_Time, height = 0.9, align = "edge", label = "GNR-PRJ", left = PRF_Time, zorder=100, log="True")
        # plt.barh(index, GNR_ACT_Time, height = 0.9, align = "edge", label = "GNR-ACT", left = np.array(PRF_Time) + np.array(GNR_PRJ_Time), zorder=100, log="True")
        # plt.barh(index, OFFCARD_Time, height = 0.9, align = "edge", label = "OFFCARD", left = np.array(PRF_Time) + np.array(GNR_PRJ_Time) + np.array(GNR_ACT_Time), zorder=100, log="True")
        plt.barh(index, PRF_Time, height=0.9, align="edge", label="Prefill", zorder=100)
        plt.barh(
            index,
            GNR_PRJ_Time,
            height=0.9,
            align="edge",
            label="GNR-PRJ",
            left=PRF_Time,
            zorder=100,
        )
        plt.barh(
            index,
            GNR_ACT_Time,
            height=0.9,
            align="edge",
            label="GNR-ACT",
            left=np.array(PRF_Time) + np.array(GNR_PRJ_Time),
            zorder=100,
        )
        plt.barh(
            index,
            OFFCARD_Time,
            height=0.9,
            align="edge",
            label="OFFCARD",
            left=np.array(PRF_Time) + np.array(GNR_PRJ_Time) + np.array(GNR_ACT_Time),
            zorder=100,
        )
        max_time = np.max(
            np.array(PRF_Time)
            + np.array(GNR_PRJ_Time)
            + np.array(GNR_ACT_Time)
            + np.array(OFFCARD_Time)
        )
        for index in range(len(Legends)):
            plt.text(
                0, index, Legends[index], ha="right", va="bottom", fontsize=fontsize
            )
            # plt.text(0.002,index, Legends[index], ha="right", va="bottom", fontsize = fontsize)

        for hard_index in range(len(Hardware)):
            plt.text(
                0 - max_time * 0.12,
                len(Model)
                * len(Ratio_P)
                * len(Batchsize)
                * len(Sparse)
                * len(Max_Token_List)
                * (hard_index + 0.5)
                + hard_index,
                Hardware[hard_index],
                fontsize=fontsize / 2,
                rotation=90,
                fontdict={"color": "red"},
            )
            # plt.text(0.0001, len(Model) * len(Ratio_P) * len(Batchsize) * len(Sparse) * len(Max_Token_List) * (hard_index+0.5)+hard_index, Hardware[hard_index], fontsize = fontsize,rotation=90, fontdict={"color": "red"})

        plt.tick_params(labelsize=10)
        plt.title("Timebreakdown " + title, fontsize=10)
        plt.xlabel("Execution Time(ms)", fontsize=10)
        a = plt.legend(fontsize=10)
        a.set_zorder(1000)
        plt.show()

    def Draw_Timebreakdown_Ratio(
        self,
        Ratio_P=[0.25],
        Batchsize=[32],
        Max_Token=[],
        Model=[],
        Hardware=[],
        Sparse=[],
    ):
        if Model == []:
            Model = [i for i in self.models.keys()]
        if Hardware == []:
            Hardware = [i for i in self.hardwares.keys()]
        if Sparse == []:
            Sparse = [i for i in self.sparses.keys()]

        self.List_Timebreakdown_Ratio(
            Ratio_P, Batchsize, Max_Token, Model, Hardware, Sparse
        )

        all_data = self.Compute_Timebreakdown(
            Ratio_P, Batchsize, Max_Token, Model, Hardware, Sparse
        )
        fig = plt.figure(dpi=300, figsize=[12, 4])
        title = ""
        hard_index = model_index = ratiop_index = batch_index = maxtoken_index = 0
        max_util = 0
        PRF_Time = []
        GNR_PRJ_Time = []
        GNR_ACT_Time = []
        OFFCARD_Time = []
        Legends = []
        index = 0

        for hard in Hardware:
            if hard == None:
                continue
            hard_info = all_data[hard]
            hard_legend = ""
            hard_title = ""
            if len(Hardware) != 1:
                hard_legend = hard + " "
            else:
                hard_title = hard + " "
            model_index = 0
            for model in Model:
                model_info = hard_info[model]
                model_legend = ""
                model_title = ""
                if len(Model) != 1:
                    model_legend = model + " "
                else:
                    model_title = model + " "
                ratiop_index = 0
                for ratiop in Ratio_P:
                    ratio_info = model_info[ratiop]
                    ratiop_legend = ""
                    ratiop_title = ""
                    if len(Ratio_P) != 1:
                        ratiop_legend = "RP=" + str(ratiop) + " "
                    else:
                        ratiop_title = "RP=" + str(ratiop) + " "
                    batch_index = 0
                    for batch in Batchsize:
                        batch_info = ratio_info[batch]
                        batchsize_legend = ""
                        batchsize_title = ""
                        # if len(Batchsize) != 1: batchsize_legend = "B=" + str(batch) + " "
                        # else: batchsize_title = "B=" + str(batch) + " "
                        sparse_index = 0
                        for sparse in Sparse:
                            sparse_info = batch_info[sparse]
                            sparse_legend = ""
                            sparse_title = ""
                            if len(Sparse) != 1:
                                sparse_legend = "S=" + str(sparse) + " "
                            else:
                                sparse_title = "S=" + str(sparse) + " "
                            if Max_Token == []:
                                Max_Token_List = [self.models[model].Max_Token]
                            else:
                                Max_Token_List = Max_Token
                            maxtoken_index = 0
                            for maxtoken in Max_Token_List:
                                maxtoken_legend = ""
                                maxtoken_title = ""
                                if len(Max_Token_List) != 1:
                                    maxtoken_legend = "L=" + str(maxtoken) + " "
                                elif Max_Token != []:
                                    maxtoken_title = "L=" + str(maxtoken) + " "

                                maxtoken_info = sparse_info[maxtoken]

                                if batch in ["Max", "max", "MAX"]:
                                    batchsize = "max({})".format(
                                        maxtoken_info["batchsize"]
                                    )
                                else:
                                    batchsize = str(batch)
                                if len(Batchsize) != 1 or batch in [
                                    "max",
                                    "Max",
                                    "MAX",
                                ]:
                                    batchsize_legend = "B=" + batchsize + " "
                                else:
                                    batchsize_title = "B=" + batchsize + " "

                                PRF_Time.append(
                                    maxtoken_info["Timebreakdown"]["PRF"]
                                    / maxtoken_info["Timebreakdown"]["Total"]
                                )
                                GNR_PRJ_Time.append(
                                    maxtoken_info["Timebreakdown"]["GNR"]["PRJ"]
                                    / maxtoken_info["Timebreakdown"]["Total"]
                                )
                                GNR_ACT_Time.append(
                                    maxtoken_info["Timebreakdown"]["GNR"]["ACT"]
                                    / maxtoken_info["Timebreakdown"]["Total"]
                                )
                                OFFCARD_Time.append(
                                    maxtoken_info["Timebreakdown"]["OFFCARD"]["Total"]
                                    / maxtoken_info["Timebreakdown"]["Total"]
                                )
                                title = (
                                    hard_title
                                    + model_title
                                    + ratiop_title
                                    + batchsize_title
                                    + sparse_title
                                    + maxtoken_title
                                )
                                Legends.append(
                                    model_legend
                                    + ratiop_legend
                                    + batchsize_legend
                                    + sparse_legend
                                    + maxtoken_legend
                                )
                                index += 1
            PRF_Time.append(0)
            GNR_PRJ_Time.append(0)
            GNR_ACT_Time.append(0)
            OFFCARD_Time.append(0)
            Legends.append("")

        index = np.arange(0, len(PRF_Time))
        fontsize = 150 / len(Legends)
        ax = plt.gca()
        ax.axes.yaxis.set_visible(False)
        plt.barh(index, PRF_Time, height=0.7, align="edge", label="Prefill")
        hbars1 = plt.barh(
            index,
            GNR_PRJ_Time,
            height=0.7,
            align="edge",
            label="GNR-PRJ",
            left=PRF_Time,
        )
        hbars2 = plt.barh(
            index,
            GNR_ACT_Time,
            height=0.7,
            align="edge",
            label="GNR-ACT",
            left=np.array(PRF_Time) + np.array(GNR_PRJ_Time),
        )
        plt.barh(
            index,
            OFFCARD_Time,
            height=0.7,
            align="edge",
            label="OFFCARD",
            left=np.array(PRF_Time) + np.array(GNR_PRJ_Time) + np.array(GNR_ACT_Time),
        )
        for index in range(len(Legends)):
            # plt.text(0.5,index, Legends[index], ha="center", va="bottom", fontsize = fontsize)
            plt.text(
                -0.05, index, Legends[index], ha="right", va="bottom", fontsize=fontsize
            )

        for hard_index in range(len(Hardware)):
            plt.text(
                -0.01,
                len(Model)
                * len(Ratio_P)
                * len(Batchsize)
                * len(Sparse)
                * len(Max_Token_List)
                * (hard_index + 0.5)
                + hard_index,
                Hardware[hard_index],
                ha="right",
                va="bottom",
                fontsize=fontsize,
                rotation=90,
            )

        ax.bar_label(
            hbars1,
            labels=[f"{x:.0%}" for x in GNR_PRJ_Time],
            fontsize=fontsize,
            color="b",
            label_type="center",
        )
        ax.bar_label(
            hbars2,
            labels=[f"{x:.0%}" for x in GNR_ACT_Time],
            fontsize=fontsize,
            color="b",
            label_type="center",
        )

        plt.tick_params(labelsize=10)
        plt.xlabel("Time Proportion", fontsize=10)
        plt.title("Timebreakdown Ratio " + title, fontsize=10)
        plt.legend(fontsize=10, loc=2)

        plt.show()

    def Find_Max_Power_2(self, A, B):
        i = -1
        while pow(2, i + 1) * A <= B:
            i += 1
        return int(pow(2, i))

    def Get_Offcard_Time(self, Trans_size_time, Card_Num, Link="NVLink"):
        Sheet_Str = ""
        if Link == "NVLink":
            if Card_Num <= 1:
                return 0
            elif Card_Num not in [2, 4, 8] and Trans_size_time[0] > 0:
                raise ValueError("Card_Num must be 2, 4 or 8!")
            Sheet_Str = str(int(Card_Num))
        else:
            Sheet_Str = Link
        size_per_trans_per_card = Trans_size_time[0]
        trans_times = Trans_size_time[1]
        if size_per_trans_per_card == 0:
            return 0
        size_sqrt_bot = int(math.log(size_per_trans_per_card, 2))
        size_bot = pow(2, size_sqrt_bot)
        if size_sqrt_bot >= len(self.all_reduce_time[Sheet_Str]):
            time_bot = self.all_reduce_time[Sheet_Str][-1] * pow(
                2, (size_sqrt_bot - len(self.all_reduce_time[Sheet_Str]) + 1)
            )
        else:
            time_bot = self.all_reduce_time[Sheet_Str][size_sqrt_bot]
        size_sqrt_up = size_sqrt_bot + 1
        size_up = pow(2, size_sqrt_up)
        if size_sqrt_up >= len(self.all_reduce_time[Sheet_Str]):
            time_up = self.all_reduce_time[Sheet_Str][-1] * pow(
                2, (size_sqrt_up - len(self.all_reduce_time[Sheet_Str]) + 1)
            )
        else:
            time_up = self.all_reduce_time[Sheet_Str][size_sqrt_up]
        total_offcard_time = time_bot + (size_per_trans_per_card - size_bot) * (
            time_up - time_bot
        ) / (size_up - size_bot)
        total_offcard_time *= trans_times
        return total_offcard_time

    def Add_Comb_Hardware(self, Param):
        comb_hard = Param
        comb_hard["PRF"]["hard"] = self.hardwares[comb_hard["PRF"]["hard"]]
        comb_hard["GNR-PRJ"]["hard"] = self.hardwares[comb_hard["GNR-PRJ"]["hard"]]
        comb_hard["GNR-ACT"]["hard"] = self.hardwares[comb_hard["GNR-ACT"]["hard"]]
        self.hardwares[comb_hard["Name"]] = Hardware(comb_hard, is_combination=True)

    def Get_Max_Batch_by_Capacity(
        self,
        hardware,
        model,
        ratiop,
        maxtoken,
        sparse,
        mag_ratio=1.2,
        max_batch=64,
        Assert=False,
    ):
        # mag_ratio = 1.2 means we give 20% larger space for hardware capacity
        assert (
            hardware in self.hardwares
        ), "No hardware named %s, please register it!" % (hardware)
        hard = self.hardwares[hardware]
        _, _, _, Capacity, _ = self.models[model].Calculate_Arithmetic(
            ratiop, 1, maxtoken, self.sparses[sparse]
        )
        Weight_Cap = Capacity["Weight"]
        # print (Weight_Cap)
        KVCache_Cap = Capacity["KVCache"]
        # print (KVCache_Cap)

        if hard.Assign == "PRF+GNR-PRJ+GNR-ACT":
            if Weight_Cap > hard.MM_Capacity * mag_ratio:
                assert Assert, "No enough capacity for weight!"
                return 0
            return min(
                int((hard.MM_Capacity * mag_ratio - Weight_Cap) / KVCache_Cap),
                max_batch,
            )
        elif hard.Assign == "PRF+GNR-PRJ, GNR-ACT":
            if Weight_Cap > hard.MM_Capacity * mag_ratio:
                assert Assert, "No enough capacity for weight!"
                return 0
            return min(int(hard.MV_Capacity * mag_ratio / KVCache_Cap), max_batch)
        elif hard.Assign == "PRF, GNR-PRJ+GNR-ACT":
            if Weight_Cap > hard.MM_Capacity * mag_ratio:
                assert Assert, "No enought capacity for weight in prefill stage!"
                return 0
            if Weight_Cap > hard.MV_Capacity * mag_ratio:
                assert Assert, "No enought capacity for weight in generation stage!"
                return 0
            return min(
                int((hard.MV_Capacity * mag_ratio - Weight_Cap) / KVCache_Cap),
                max_batch,
            )
        else:
            assert True, "Hardware Assign %s is not supported!" % (hard.Assign)

    def Get_Max_Batch(
        self,
        hardware,
        model,
        ratiop,
        maxtoken,
        sparse,
        pipeline=1,
        max_FPRT=3,
        min_token_per_sec=20,
        first_col=True,
        output=True,
        round_2=False,
    ):
        Max_Batch_Capacity = 0
        Max_Batch_FPRT = 0
        Max_Batch_Token_Per_Sec = 0
        if first_col and output:
            print("\033[1;31m*******List_Max_Batch*******\033[0m")
            print(
                "\033[1;31m%-15s %-20s %-8s %-10s %-12s %-10s %-10s %-11s %-14s %-11s\033[0m"
                % (
                    "Model",
                    "Hardware",
                    "Ratio_P",
                    "Max_Token",
                    "Attention",
                    "Sparse",
                    "Cap-Bound",
                    "FPRT-Bound",
                    "Token/s-Bound",
                    "Min",
                )
            )
        Max_Batch_Capacity = self.Get_Max_Batch_by_Capacity(
            hardware,
            model,
            ratiop,
            maxtoken,
            sparse,
            mag_ratio=1.2,
            Assert=True,
            max_batch=1e10,
        )

        attention = "MHA"
        if self.models[model].Multi_Query:
            attention = "MQA"
        elif self.models[model].Grouped_Query:
            attention = "GQA-" + str(self.models[model].Grouped_Num)

        pre_batch = 0
        pre_FPRT = 0
        current_FPRT = 0
        while 1:
            all_data = self.Compute_Timebreakdown(
                Ratio_P=[ratiop],
                Batchsize=[pre_batch + 1],
                Max_Token=[maxtoken],
                Model=[model],
                Hardware=[hardware],
                Sparse=[sparse],
                Pipeline_Stage=1,
            )
            maxtoken_info = all_data[hardware][model][ratiop][pre_batch + 1][sparse][
                maxtoken
            ]
            current_FPRT = (
                maxtoken_info["Timebreakdown"]["PRF"]
                + maxtoken_info["Timebreakdown"]["OFFCARD"]["PRF"]["Total"]
            )
            # print ("current_FPRT:", current_FPRT, "batch", pre_batch, "max_FPRT:", max_FPRT)
            if pre_FPRT >= max_FPRT or (
                pre_FPRT < max_FPRT and current_FPRT >= max_FPRT
            ):
                Max_Batch_FPRT = pre_batch
                break
            pre_batch += 1
            pre_FPRT = current_FPRT

        pre_batch = 0
        pre_token_per_sec = 100
        current_token_per_sec = 0
        while 1:
            all_data = self.Compute_Timebreakdown(
                Ratio_P=[ratiop],
                Batchsize=[pre_batch + 1],
                Max_Token=[maxtoken],
                Model=[model],
                Hardware=[hardware],
                Sparse=[sparse],
                Pipeline_Stage=1,
            )
            maxtoken_info = all_data[hardware][model][ratiop][pre_batch + 1][sparse][
                maxtoken
            ]
            s_per_out_token = (
                pipeline
                * (
                    maxtoken_info["Timebreakdown"]["Total"]
                    - maxtoken_info["Timebreakdown"]["PRF"]
                    - maxtoken_info["Timebreakdown"]["OFFCARD"]["PRF"]["Total"]
                )
                / (maxtoken * (1 - ratiop))
            )
            # print (s_per_out_token)
            current_token_per_sec = 1 / s_per_out_token
            if pre_token_per_sec <= min_token_per_sec or (
                pre_token_per_sec > min_token_per_sec
                and current_token_per_sec <= min_token_per_sec
            ):
                Max_Batch_Token_Per_Sec = pre_batch
                break
            pre_batch += 1
            pre_token_per_sec = current_token_per_sec
        if round_2:
            Max_Batch_Capacity = (
                self.Find_Max_Power_2(1, Max_Batch_Capacity)
                if Max_Batch_Capacity >= 1
                else 0
            )
            Max_Batch_FPRT = (
                self.Find_Max_Power_2(1, Max_Batch_FPRT) if Max_Batch_FPRT >= 1 else 0
            )
            Max_Batch_Token_Per_Sec = (
                self.Find_Max_Power_2(1, Max_Batch_Token_Per_Sec)
                if Max_Batch_Token_Per_Sec >= 1
                else 0
            )
        if output:
            print(
                "%-15s %-20s %-8s %-10s %-12s %-10s %-10s %-11s %-14s %-11s"
                % (
                    model,
                    hardware,
                    ratiop,
                    maxtoken,
                    attention,
                    sparse,
                    Max_Batch_Capacity,
                    Max_Batch_FPRT,
                    Max_Batch_Token_Per_Sec,
                    min(Max_Batch_Capacity, Max_Batch_FPRT, Max_Batch_Token_Per_Sec),
                )
            )
        return Max_Batch_Capacity, Max_Batch_FPRT, Max_Batch_Token_Per_Sec

    def Evaluate_Dataset(
        self,
        DataSet=[],
        Model=[],
        Hardware=[],
        Batch_Policy=[],
        Batchsize=[],
        Pipeline_Stage=1,
    ):
        # DataSet = [[input_length_0, output_length_0], [input_length_1, output_length_1], ...]
        # Batch_Policy = "Sequential" or "Input_Nearest"
        # Output: Compute/Communication Amount/Time of each part, Wasted Time of Prefill/Generation, Total Utilization
        pass

    def Calc_SLO(
        self,
        Query_Per_Second,
        Deadline,
        Execution_Time_Per_Batch,
        Prefill_Time_Per_Batch,
        batch,
    ):
        exp_num = int(2e5)
        Arrival_Interval_List = np.random.exponential(1 / Query_Per_Second, exp_num)
        Arrival_Time_List = [round(Arrival_Interval_List[0], 2)]
        Issue_Time_List = []
        Batch_List = []
        Wait_Time_List = []
        for i in np.arange(1, exp_num):
            Arrival_Time_List.append(
                round(Arrival_Time_List[i - 1] + Arrival_Interval_List[i], 2)
            )

        Time_Stamp = Execution_Time_Per_Batch
        Traverse_Index = 0

        while Traverse_Index < exp_num:
            Index_in_Batch = 0
            while (
                Index_in_Batch < batch
                and Traverse_Index < exp_num
                and Arrival_Time_List[Traverse_Index] < Time_Stamp
            ):
                Issue_Time_List.append(Time_Stamp)
                Wait_Time_List.append(
                    Time_Stamp
                    + Prefill_Time_Per_Batch
                    - Arrival_Time_List[Traverse_Index]
                )
                Index_in_Batch += 1
                Traverse_Index += 1
            Batch_List.append(Index_in_Batch)
            Time_Stamp += Execution_Time_Per_Batch
        Average_Wait_Time = np.average(Wait_Time_List)
        If_Time_Out = np.array(Wait_Time_List) > Deadline
        Time_Out_Ratio = np.sum(If_Time_Out) / len(Wait_Time_List)
        return Time_Out_Ratio

    def Evaluate_SLO_Attainment(
        self,
        Query_Per_Second,
        Dead_Time=[],
        Ratio_P=[0.25],
        Batchsize=[32],
        Max_Token=[],
        Model=[],
        Hardware=[],
        Sparse=[],
        Pipeline_Stage=1,
        first_col=True,
    ):
        if Model == []:
            Model = [i for i in self.models.keys()]
        if Hardware == []:
            Hardware = [i for i in self.hardwares.keys()]
        if Sparse == []:
            Sparse = [i for i in self.sparses.keys()]

        if Dead_Time == []:  # Do not need to output SLO Attainment
            Dead_Time = [0]

        all_data = self.Compute_Timebreakdown(
            Ratio_P, Batchsize, Max_Token, Model, Hardware, Sparse, Pipeline_Stage
        )
        if first_col:
            print("\033[1;31m*******List_Timebreakdown*******\033[0m")
            # print("\033[1;31m%-8s%-20s%-12s%-8s%-7s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-12s%-13s%-10s%-10s%-10s%-15s\033[0m"% \
            print(
                "\033[1;31m%-8s%-20s%-12s%-8s%-7s%-10s%-10s%-10s%-10s%-15s%-15s%-15s%-15s\033[0m"
                % (
                    "Model",
                    "Hardware",
                    "Pipe_Stage",
                    "Ratio_P",
                    "Batch",
                    "Max_Token",
                    "Sparse",
                    "Dead_Time",  # "Comp-P", "Arith-P", "Time-P", "Util-P", "Comp-GP", "Arith-P", "Time-GP", "Util-GP", "Comp-GA", "Arith-P", "Time-GA", "Util-GA", "Comp-Ttl", "Time-Offcard", "Time-Ttl", "Util-Ttl", "Token/Sec", "Token/Sec/Batch" ))
                    # "Comp-P", "Arith-P", "Time-P", "Util-P", "Comp-GP", "Arith-P", "Time-GP", "Util-GP", "Comp-GA", "Arith-P", "Time-GA", "Util-GA", "Comp-Ttl", "Time-Comm", "Time-Ttl", "Util-Ttl", "OutToken/Sec", "AllToken/Sec"))
                    "Time-Ttl",
                    "OutToken/Sec",
                    "AllToken/Sec",
                    "Avg_Wait",
                    "SLO_Attn",
                )
            )

        for hard in Hardware:
            if hard == None:
                continue
            hard_info = all_data[hard]
            for model in Model:
                model_info = hard_info[model]
                for ratiop in Ratio_P:
                    ratio_info = model_info[ratiop]
                    for batch in Batchsize:
                        batch_info = ratio_info[batch]
                        batch_name = str(batch)
                        for sparse in Sparse:
                            sparse_info = batch_info[sparse]
                            if Max_Token == []:
                                Max_Token_List = [self.models[model].Max_Token]
                            else:
                                Max_Token_List = Max_Token
                            for maxtoken in Max_Token_List:
                                Parallel_str = str(Pipeline_Stage)
                                Throughput_Ratio = Pipeline_Stage
                                maxtoken_info = sparse_info[maxtoken]
                                Execution_Time_Per_Batch = maxtoken_info[
                                    "Timebreakdown"
                                ]["Total"]
                                Prefill_Time_Per_Batch = maxtoken_info["Timebreakdown"][
                                    "PRF"
                                ]
                                # Execution_Time_Per_Batch = 10
                                # print("Execution_Time:", Execution_Time_Per_Batch)

                                exp_num = int(2e5)
                                Arrival_Interval_List = np.random.exponential(
                                    1 / Query_Per_Second, exp_num
                                )
                                Arrival_Time_List = [round(Arrival_Interval_List[0], 2)]
                                Issue_Time_List = []
                                Batch_List = []
                                Wait_Time_List = []
                                for i in np.arange(1, exp_num):
                                    Arrival_Time_List.append(
                                        round(
                                            Arrival_Time_List[i - 1]
                                            + Arrival_Interval_List[i],
                                            2,
                                        )
                                    )

                                Time_Stamp = Execution_Time_Per_Batch
                                Traverse_Index = 0

                                if batch_name in ["max", "Max", "MAX"]:
                                    batch = maxtoken_info["batchsize"]
                                while Traverse_Index < exp_num:
                                    Index_in_Batch = 0
                                    while (
                                        Index_in_Batch < batch
                                        and Traverse_Index < exp_num
                                        and Arrival_Time_List[Traverse_Index]
                                        < Time_Stamp
                                    ):
                                        Issue_Time_List.append(Time_Stamp)
                                        Wait_Time_List.append(
                                            Time_Stamp
                                            + Prefill_Time_Per_Batch
                                            - Arrival_Time_List[Traverse_Index]
                                        )
                                        Index_in_Batch += 1
                                        Traverse_Index += 1
                                    Batch_List.append(Index_in_Batch)
                                    Time_Stamp += Execution_Time_Per_Batch

                                Average_Wait_Time = np.average(Wait_Time_List)

                                # print("Arrival_Interval_List:", Arrival_Interval_List)
                                # print("Arrival_Time_List:", Arrival_Time_List)
                                # print("Issue_Time_List:", Issue_Time_List)
                                # for i in range(exp_num):
                                # print(Arrival_Time_List[i]," ", Issue_Time_List[i], " ", Wait_Time_List[i], " ", If_Time_Out[i])
                                # print("Batch_List:", Batch_List)
                                # print("Average_Wait_Time:", Average_Wait_Time)
                                # print("Time_Out_Ratio:", Time_Out_Ratio)

                                if batch_name in ["Max", "max", "MAX"]:
                                    batch_name_print = "max({})".format(batch)
                                else:
                                    batch_name_print = "{}".format(batch)
                                for slo in Dead_Time:
                                    If_Time_Out = np.array(Wait_Time_List) > slo
                                    Time_Out_Ratio = np.sum(If_Time_Out) / len(
                                        Wait_Time_List
                                    )
                                    print(
                                        "%-8s%-20s%-12s%-8s%-7s%-10s%-10s%-10d%-10.2f%-15.2f%-15.2f%-15.2f%-15.2f"
                                        % (
                                            model,
                                            hard,
                                            Parallel_str,
                                            ratiop,
                                            batch_name_print,
                                            maxtoken,
                                            sparse,
                                            slo,
                                            maxtoken_info["Timebreakdown"][
                                                "Total"
                                            ],  # Time-Ttl
                                            batch
                                            * maxtoken
                                            * (1 - ratiop)
                                            * Throughput_Ratio
                                            / (
                                                maxtoken_info["Timebreakdown"]["Total"]
                                            ),  # OutToken/Sec
                                            batch
                                            * maxtoken
                                            * Throughput_Ratio
                                            / (
                                                maxtoken_info["Timebreakdown"]["Total"]
                                            ),  #  AllToken/Sec
                                            Average_Wait_Time,  # Average_Wait_Time
                                            1 - Time_Out_Ratio,
                                        )
                                    )  # SLO_Attn
                                    # pass
                                    # print("%-8s%-20s%-12s%-8.2f%-7d%-10d%-10s%-10.2f%-11.2f%-11.2f%-10.4f%-10.2f%-12.2f%-12.2f%-10.4f%-10.2f%-12.2f%-12.2f%-10.4f%-13.2f%-13.2f%-10.2f%-10s%-15.2f%-15.2f" % \
                                    #     (model, hard, Parallel_str,ratiop, batch, maxtoken, sparse,
                                    #     maxtoken_info["Compute"]["PRF"]["PRJ"] + maxtoken_info["Compute"]["PRF"]["ACT"], # Comp-P
                                    #     maxtoken_info["Timebreakdown"]["PRF"], # CompTime-P
                                    #     maxtoken_info["Timebreakdown"]["OFFCARD"]["PRF"]["Total"], # CommTime-P
                                    #     maxtoken_info["Timebreakdown"]["PRF"] + maxtoken_info["Timebreakdown"]["OFFCARD"]["PRF"]["Total"], # Time-P

                                    #     maxtoken_info["Compute"]["GNR"]["PRJ"], # Comp-GP
                                    #     maxtoken_info["Timebreakdown"]["GNR"]["PRJ"], # CompTime-GP
                                    #     maxtoken_info["Timebreakdown"]["OFFCARD"]["GNR-PRJ"]["Total"], # CommTime-GP
                                    #     maxtoken_info["Timebreakdown"]["GNR"]["PRJ"] + maxtoken_info["Timebreakdown"]["OFFCARD"]["GNR-PRJ"]["Total"], # Time-GP

                                    #     maxtoken_info["Compute"]["GNR"]["ACT"], # Comp-GA
                                    #     maxtoken_info["Timebreakdown"]["GNR"]["ACT"], # CompTime-GA
                                    #     maxtoken_info["Timebreakdown"]["OFFCARD"]["GNR-ACT"]["Total"], # CommTime-GA
                                    #     maxtoken_info["Timebreakdown"]["GNR"]["ACT"] + maxtoken_info["Timebreakdown"]["OFFCARD"]["GNR-ACT"]["Total"], # Time-GA

                                    #     maxtoken_info["Timebreakdown"]["PRF"] + maxtoken_info["Timebreakdown"]["GNR"]["PRJ"] + maxtoken_info["Timebreakdown"]["GNR"]["ACT"],# CompTime-Ttl
                                    #     maxtoken_info["Timebreakdown"]["OFFCARD"]["Total"],# CommTime-Ttl
                                    #     maxtoken_info["Timebreakdown"]["Total"],# Time-Ttl

                                    #     str(int(maxtoken_info["Util"]["Total"]*1000)/1000) if self.hardwares[hard].Form_Type=="Homo" else "NA",# Util-Ttl
                                    #     batch * maxtoken * (1-ratiop) * Throughput_Ratio / (maxtoken_info["Timebreakdown"]["Total"]),# OutToken/Sec
                                    #     batch * maxtoken * Throughput_Ratio / (maxtoken_info["Timebreakdown"]["Total"]))) # AllToken/Sec

    def Draw_token_per_second_breakdown(
        self,
        Ratio_P=[0.25],
        Batchsize=[32],
        Max_Token=[4096],
        Model=["GPT3.5"],
        Hardware=[],
        Sparse=["default"],
        Parallel="Model",
        Latency_QPS="Latency",
        precentage=False,
        Pipeline_Stage=1,
        figsize=[6.4, 2.8],
        show_cost=False,
    ):
        all_data = self.Compute_Timebreakdown(
            Ratio_P, Batchsize, Max_Token, Model, Hardware, Sparse, Pipeline_Stage
        )

        assert len(Model) == 1, "draw different model in different figures"
        assert len(Ratio_P) == 1, "ratio P variation is lower priority"

        # group = batch or Max_token
        assert (
            min(len(Batchsize), len(Max_Token)) == 1
        ), "either explore batch or seq_len"
        group = Batchsize if len(Batchsize) > 1 else Max_Token
        # per group = HW or sparse
        assert min(len(Hardware), len(Sparse)) == 1, "either explore batch or seq_len"
        per_group = Hardware if len(Hardware) > 1 else Sparse

        all_data_reorg = {}
        breakdown = {
            "FPRT": [],
            "PRE": {"Comp": [], "Comm": []},
            "GP": {"Comp": [], "Comm": []},
            "GA": {"Comp": [], "Comm": []},
            "Util": {"Tot": [], "GP": [], "GA": []},
            "Cost": [],
            "Speedup": [],
            "Tot": [],
        }
        for g in group:
            batch = g if len(Batchsize) > 1 else Batchsize[0]
            maxtoken = g if len(Max_Token) > 1 else Max_Token[0]
            all_data_reorg[g] = {}
            base = 0
            batch_in_group = []
            for pg in per_group:
                hard = pg if len(Hardware) > 1 else Hardware[0]
                sparse = pg if len(Sparse) > 1 else Sparse[0]

                all_data_reorg[g][pg] = all_data[hard][Model[0]][Ratio_P[0]][batch][
                    sparse
                ][maxtoken]
                batchsize = all_data_reorg[g][pg]["batchsize"]
                batch_in_group.extend([str(batchsize), ","])
                self.cost_model.init_hardware(self.hardwares[hard])

                breakdown["FPRT"].append(
                    all_data_reorg[g][pg]["Timebreakdown"]["PRF"]
                    + all_data_reorg[g][pg]["Timebreakdown"]["OFFCARD"]["PRF"]["Total"]
                )
                if Latency_QPS == "Latency":  # ms/token/quary
                    breakdown["Tot"].append(
                        (
                            all_data_reorg[g][pg]["Timebreakdown"]["GNR"]["PRJ"]
                            + all_data_reorg[g][pg]["Timebreakdown"]["OFFCARD"][
                                "GNR-PRJ"
                            ]["Total"]
                            + all_data_reorg[g][pg]["Timebreakdown"]["GNR"]["ACT"]
                            + all_data_reorg[g][pg]["Timebreakdown"]["OFFCARD"][
                                "GNR-ACT"
                            ]["Total"]
                        )
                        / maxtoken
                    )
                    normalization = breakdown["Tot"][-1] if precentage == True else 1
                    breakdown["PRE"]["Comp"].append(0.0)
                    breakdown["PRE"]["Comm"].append(0.0)
                    breakdown["GP"]["Comp"].append(
                        all_data_reorg[g][pg]["Timebreakdown"]["GNR"]["PRJ"]
                        / maxtoken
                        / normalization
                        * 1000
                    )
                    breakdown["GP"]["Comm"].append(
                        all_data_reorg[g][pg]["Timebreakdown"]["OFFCARD"]["GNR-PRJ"][
                            "Total"
                        ]
                        / maxtoken
                        / normalization
                        * 1000
                    )
                    breakdown["GA"]["Comp"].append(
                        all_data_reorg[g][pg]["Timebreakdown"]["GNR"]["ACT"]
                        / maxtoken
                        / normalization
                        * 1000
                    )
                    breakdown["GA"]["Comm"].append(
                        all_data_reorg[g][pg]["Timebreakdown"]["OFFCARD"]["GNR-ACT"][
                            "Total"
                        ]
                        / maxtoken
                        / normalization
                        * 1000
                    )
                else:  # quary latency = 1/qps
                    breakdown["Tot"].append(
                        (all_data_reorg[g][pg]["Timebreakdown"]["Total"]) / batchsize
                    )
                    normalization = breakdown["Tot"][-1] if precentage == True else 1
                    breakdown["PRE"]["Comp"].append(
                        all_data_reorg[g][pg]["Timebreakdown"]["PRF"]
                        / batchsize
                        / normalization
                    )
                    breakdown["PRE"]["Comm"].append(
                        all_data_reorg[g][pg]["Timebreakdown"]["OFFCARD"]["PRF"][
                            "Total"
                        ]
                        / batchsize
                        / normalization
                    )
                    breakdown["GP"]["Comp"].append(
                        all_data_reorg[g][pg]["Timebreakdown"]["GNR"]["PRJ"]
                        / batchsize
                        / normalization
                    )
                    breakdown["GP"]["Comm"].append(
                        all_data_reorg[g][pg]["Timebreakdown"]["OFFCARD"]["GNR-PRJ"][
                            "Total"
                        ]
                        / batchsize
                        / normalization
                    )
                    breakdown["GA"]["Comp"].append(
                        all_data_reorg[g][pg]["Timebreakdown"]["GNR"]["ACT"]
                        / batchsize
                        / normalization
                    )
                    breakdown["GA"]["Comm"].append(
                        all_data_reorg[g][pg]["Timebreakdown"]["OFFCARD"]["GNR-ACT"][
                            "Total"
                        ]
                        / batchsize
                        / normalization
                    )
                breakdown["Util"]["Tot"].append(all_data_reorg[g][pg]["Util"]["Total"])
                breakdown["Util"]["GP"].append(
                    all_data_reorg[g][pg]["Util"]["GNR"]["PRJ"]
                )
                breakdown["Util"]["GA"].append(
                    all_data_reorg[g][pg]["Util"]["GNR"]["ACT"]
                )
                breakdown["Cost"].append(
                    100
                    * 1000
                    * self.cost_model.cal_total_cost(
                        maxtoken
                        * batchsize
                        * Pipeline_Stage
                        / (all_data_reorg[g][pg]["Timebreakdown"]["Total"])
                    )
                )
                base = breakdown["Tot"][-1] if base == 0 else base
                breakdown["Speedup"].append(
                    "{:.2f}x".format(base / breakdown["Tot"][-1])
                )
            breakdown["FPRT"].append(0.0)
            breakdown["PRE"]["Comp"].append(0.0)
            breakdown["PRE"]["Comm"].append(0.0)
            breakdown["GP"]["Comp"].append(0.0)
            breakdown["GP"]["Comm"].append(0.0)
            breakdown["GA"]["Comp"].append(0.0)
            breakdown["GA"]["Comm"].append(0.0)
            breakdown["Util"]["Tot"].append(0.0)
            breakdown["Util"]["GP"].append(0.0)
            breakdown["Util"]["GA"].append(0.0)
            breakdown["Cost"].append(0.0)
            breakdown["Speedup"].append("")
            breakdown["Tot"].append(0.0)

        ## draw fig
        fontsize = 5
        index = np.arange(0, len(group) * (len(per_group) + 1))

        if not show_cost:
            fig = plt.figure(dpi=300, figsize=figsize)
            ax = plt.gca()
            ax0 = ax
        else:
            fig, (ax0, ax) = plt.subplots(
                nrows=2,
                ncols=1,
                sharex=True,
                squeeze=True,
                height_ratios=[0.2, 0.8],
                dpi=300,
                figsize=figsize,
            )

            for g in range(len(group)):
                ax0.plot(
                    index[
                        g * (len(per_group) + 1) : (g + 1) * (len(per_group) + 1) - 1
                    ],
                    breakdown["Cost"][
                        g * (len(per_group) + 1) : (g + 1) * (len(per_group) + 1) - 1
                    ],
                    marker="^",
                    color="b",
                )
                for i in index[
                    g * (len(per_group) + 1) : (g + 1) * (len(per_group) + 1) - 1
                ]:
                    ax0.annotate(
                        "{:.2f}".format(breakdown["Cost"][i]),
                        (i + 0.1, breakdown["Cost"][i]),
                        fontsize=fontsize,
                        color="black",
                        rotation=45,
                    )
            ax0.set_xticklabels([], fontsize=0)
            ax0.tick_params(axis="x", which="major", labelsize=0, length=0)
            ax0.set_yticklabels([], fontsize=0)
            ax0.set_ylabel("\xa2-RMB/k-token", fontsize=fontsize)

        hbars1 = ax.bar(
            index,
            breakdown["PRE"]["Comp"],
            width=0.5,
            align="center",
            label="PRE-gemm",
            color="maroon",
        )
        acc = np.array(breakdown["PRE"]["Comp"])
        hbars2 = ax.bar(
            index,
            breakdown["PRE"]["Comm"],
            width=0.5,
            align="center",
            label="PRE-comm",
            color="darkgrey",
            bottom=acc,
        )
        acc += np.array(breakdown["PRE"]["Comm"])
        hbars3 = ax.bar(
            index,
            breakdown["GP"]["Comp"],
            width=0.5,
            align="center",
            label="GP-gemm",
            color="royalblue",
            bottom=acc,
        )
        acc += np.array(breakdown["GP"]["Comp"])
        hbars4 = ax.bar(
            index,
            breakdown["GP"]["Comm"],
            width=0.5,
            align="center",
            label="GP-comm",
            color="darkgrey",
            bottom=acc,
        )
        acc += np.array(breakdown["GP"]["Comm"])
        hbars5 = ax.bar(
            index,
            breakdown["GA"]["Comp"],
            width=0.5,
            align="center",
            label="GA-gemm",
            color="darkorange",
            bottom=acc,
        )
        acc += np.array(breakdown["GA"]["Comp"])
        hbars6 = ax.bar(
            index,
            breakdown["GA"]["Comm"],
            width=0.5,
            align="center",
            label="GA-comm",
            color="gray",
            bottom=acc,
        )

        ax.set_xticks(index, (per_group + [""]) * len(group), minor=False, rotation=40)
        group_name = [
            "{} = {}".format("batch" if len(Batchsize) > 1 else "seq", g) for g in group
        ]
        if len(Batchsize) > 1 and Batchsize[-1] in ["MAX", "max", "Max"]:
            group_name[-1] = "max batch = " + "".join(batch_in_group[:-1])
        ax.set_xticks(
            np.arange(0, len(group)) * (len(per_group) + 1) + len(per_group) / 2 - 0.5,
            group_name,
            minor=True,
        )

        title = "{}: {} {:.2%} input token and ".format(
            Model[0],
            (
                "latency (ms/token/quary) breakdown \n w/ "
                if Latency_QPS == "Latency"
                else "throughput (1/qps) breakdown \n w/ "
            ),
            Ratio_P[0],
        )
        if len(Batchsize) > 1:
            title += "{}={}".format("Seq", Max_Token[0])
        else:
            title += "{}={}".format("Batch", Batchsize[0])
        ax0.set_title(title, fontsize=fontsize)

        if precentage == False:
            ax.bar_label(
                hbars6, breakdown["Speedup"], fontsize=fontsize, color="b", rotation=45
            )
            ax.bar_label(
                hbars1,
                [
                    "Util: {:.1%}".format(x) if x != 0 else ""
                    for x in breakdown["Util"]["Tot"]
                ],
                fontsize=fontsize - 1,
                color="black",
                rotation=90,
                label_type="edge",
            )
        else:
            ax.bar_label(
                hbars2,
                ["{:.1%}".format(x) if x != 0 else "" for x in breakdown["GP"]["Comp"]],
                fontsize=fontsize - 1,
                color="black",
                rotation=90,
                label_type="edge",
            )
            #    ax.bar_label(hbars3, ["{:.1%}".format(x) if x != 0 else "" for x in breakdown["GP"]["Comp"]], fontsize = fontsize - 1, color = 'black', rotation = 90, label_type = 'center')
            ax.bar_label(
                hbars3,
                ["{:.1%}".format(x) if x != 0 else "" for x in breakdown["GP"]["Comm"]],
                fontsize=fontsize - 1,
                color="black",
                rotation=90,
                label_type="edge",
            )
            ax.bar_label(
                hbars4,
                ["{:.1%}".format(x) if x != 0 else "" for x in breakdown["GA"]["Comp"]],
                fontsize=fontsize - 1,
                color="black",
                rotation=90,
                label_type="edge",
            )
        #    ax.bar_label(hbars5, ["{:.1%}".format(x) if x != 0 else "" for x in breakdown["GA"]["Comp"]], fontsize = fontsize - 1, color = 'black', rotation = 90, label_type = 'center')

        ax.legend(fontsize=fontsize)

        ax.tick_params(axis="x", which="major", labelsize=fontsize, length=0)
        ax.tick_params(
            axis="x", which="minor", labelsize=fontsize + 0.5, length=30, width=0
        )
        ax.tick_params(axis="y", labelsize=fontsize)
        ax.set_ylabel(
            "ms/token/query" if Latency_QPS == "Latency" else "s/query",
            fontsize=fontsize,
        )
        fig.tight_layout(h_pad=-1.0)
        plt.show()

    def Read_Measurement(
        self,
        file="measure_vs_roofline.csv",
        Latency_QPS="Latency",
        precentage=False,
        communication=True,
        utilization_opt_stage=False,
    ):
        self.measure = {}
        breakdown = {
            "FPRT": [],
            "PRE": {"Comp": [], "Comm": [], "Ele": []},
            "GP": {"Comp": [], "Comm": [], "Ele": []},
            "GA": {"Comp": [], "Comm": [], "Ele": []},
            "Util": {"Tot": [], "GP": [], "GA": []},
            "Cost": [],
            "Speedup": [],
            "Tot": [],
            "Name": [],
        }
        # all_data[hard][Model[0]][Ratio_P[0]][batch][sparse][maxtoken]
        record = False
        with open(file, "r") as f:
            for line in f:
                line = line.split(",")
                if line[0] == "Multi-Iter":
                    record = True
                    model = line[1]
                    hard = line[5] + "V100"  # [FIXME]
                elif line[0] != "" or line[10] == "":
                    record = False
                if record:
                    batch = int(line[6])
                    in_len = int(line[7])
                    out_len = int(line[8])
                    ratiop = in_len / (in_len + out_len)
                    maxtoken = in_len + out_len
                    name = "batch={},in={},out={}".format(batch, in_len, out_len)
                    rf_data = self.Compute_Timebreakdown(
                        [ratiop], [batch], [maxtoken], [model], [hard], ["default"]
                    )
                    # self.List_Timebreakdown_Ratio([ratiop], [batch], [maxtoken], [model], [hard], ["default"])
                    rf_data = rf_data[hard][model][ratiop][batch]["default"][maxtoken]
                    self.measure[name] = {
                        "measure": {},
                        "roofline": {},
                        "batch": batch,
                        "maxtoken": maxtoken,
                        "ratiop": ratiop,
                    }

                    breakdown["Name"].extend([name, "", ""])
                    self.measure[name]["measure"]["PRE_COMP"] = (
                        float(line[12]) + float(line[14])
                    ) / batch
                    self.measure[name]["roofline"]["PRE_COMP"] = (
                        rf_data["Timebreakdown"]["PRF"] / batch
                        if communication == True
                        else 0.0
                    )
                    self.measure[name]["measure"]["PRE_COMM"] = (
                        float(line[10]) + float(line[11])
                    ) / batch
                    self.measure[name]["roofline"]["PRE_COMM"] = (
                        rf_data["Timebreakdown"]["OFFCARD"]["PRF"]["Total"] / batch
                        if communication == True
                        else 0.0
                    )
                    self.measure[name]["measure"]["PRE_ELE"] = (
                        float(line[15])
                        + float(line[16])
                        + float(line[17])
                        + float(line[18])
                    ) / batch
                    self.measure[name]["roofline"]["PRE_ELE"] = 0.0
                    if Latency_QPS == "Latency":
                        breakdown["PRE"]["Comp"].extend([0.0, 0.0, 0])
                        breakdown["PRE"]["Comm"].extend([0.0, 0.0, 0])
                        breakdown["PRE"]["Ele"].extend([0.0, 0.0, 0])
                        normalization = out_len
                    else:
                        breakdown["PRE"]["Comp"].extend(
                            [
                                self.measure[name]["measure"]["PRE_COMP"],
                                self.measure[name]["roofline"]["PRE_COMP"],
                                0,
                            ]
                        )
                        breakdown["PRE"]["Comm"].extend(
                            [
                                self.measure[name]["measure"]["PRE_COMM"],
                                self.measure[name]["roofline"]["PRE_COMM"],
                                0,
                            ]
                        )
                        breakdown["PRE"]["Ele"].extend(
                            [
                                self.measure[name]["measure"]["PRE_ELE"],
                                self.measure[name]["roofline"]["PRE_ELE"],
                                0,
                            ]
                        )
                        normalization = batch

                    self.measure[name]["measure"]["GP_COMP"] = (
                        float(line[22]) - float(line[12])
                    ) / normalization
                    self.measure[name]["roofline"]["GP_COMP"] = (
                        rf_data["Timebreakdown"]["GNR"]["PRJ"] / normalization
                    )
                    breakdown["GP"]["Comp"].extend(
                        [
                            self.measure[name]["measure"]["GP_COMP"],
                            self.measure[name]["roofline"]["GP_COMP"],
                            0,
                        ]
                    )
                    self.measure[name]["measure"]["GP_COMM"] = (
                        float(line[21]) - float(line[11])
                    ) / normalization
                    self.measure[name]["roofline"]["GP_COMM"] = (
                        rf_data["Timebreakdown"]["OFFCARD"]["GNR-PRJ"]["Total"]
                        / normalization
                        if communication == True
                        else 0.0
                    )
                    breakdown["GP"]["Comm"].extend(
                        [
                            self.measure[name]["measure"]["GP_COMM"],
                            self.measure[name]["roofline"]["GP_COMM"],
                            0,
                        ]
                    )
                    self.measure[name]["measure"]["GP_ELE"] = (
                        float(line[26])
                        + float(line[27])
                        + float(line[28])
                        - float(line[16])
                        - float(line[17])
                        - float(line[18])
                    ) / normalization
                    self.measure[name]["roofline"]["GP_ELE"] = 0.0
                    breakdown["GP"]["Ele"].extend(
                        [
                            self.measure[name]["measure"]["GP_ELE"],
                            self.measure[name]["roofline"]["GP_ELE"],
                            0,
                        ]
                    )

                    self.measure[name]["measure"]["GA_COMP"] = (
                        float(line[24]) - float(line[14])
                    ) / normalization
                    self.measure[name]["roofline"]["GA_COMP"] = (
                        rf_data["Timebreakdown"]["GNR"]["ACT"] / normalization
                    )
                    breakdown["GA"]["Comp"].extend(
                        [
                            self.measure[name]["measure"]["GA_COMP"],
                            self.measure[name]["roofline"]["GA_COMP"],
                            0,
                        ]
                    )
                    self.measure[name]["measure"]["GA_COMM"] = 0.0
                    self.measure[name]["roofline"]["GA_COMM"] = (
                        rf_data["Timebreakdown"]["OFFCARD"]["GNR-ACT"]["Total"]
                        / normalization
                        if communication == True
                        else 0.0
                    )
                    breakdown["GA"]["Comm"].extend(
                        [
                            self.measure[name]["measure"]["GA_COMM"],
                            self.measure[name]["roofline"]["GA_COMM"],
                            0,
                        ]
                    )
                    self.measure[name]["measure"]["GA_ELE"] = (
                        float(line[25]) - float(line[15])
                    ) / normalization
                    self.measure[name]["roofline"]["GA_ELE"] = 0.0
                    breakdown["GA"]["Ele"].extend(
                        [
                            self.measure[name]["measure"]["GA_ELE"],
                            self.measure[name]["roofline"]["GA_ELE"],
                            0,
                        ]
                    )

                    tm, tr = 0.0, 0.0
                    tm_gp, tm_ga, tr_gp, tr_ga = 0, 0, 0, 0
                    for k in self.measure[name]["measure"].keys():
                        if Latency_QPS == "Latency" and "PRE" in k:
                            continue
                        tm += self.measure[name]["measure"][k]
                        tr += self.measure[name]["roofline"][k]
                        if "GP" in k:
                            tm_gp += self.measure[name]["measure"][k]
                            tr_gp += self.measure[name]["roofline"][k]
                        if "GA" in k:
                            tm_ga += self.measure[name]["measure"][k]
                            tr_ga += self.measure[name]["roofline"][k]
                    (
                        self.measure[name]["measure"]["Tot"],
                        self.measure[name]["roofline"]["Tot"],
                    ) = (tm, tr)
                    breakdown["Tot"].extend([tm, tr, 0])

                    self.measure[name]["measure"]["Util_TOT"] = (
                        rf_data["Util"]["Total"] * tr / tm
                    )
                    t_noele = (
                        tm
                        - self.measure[name]["measure"]["PRE_ELE"]
                        - self.measure[name]["measure"]["GP_ELE"]
                        - self.measure[name]["measure"]["GA_ELE"]
                    )
                    t_nocomm = (
                        t_noele
                        - self.measure[name]["measure"]["PRE_COMM"]
                        - self.measure[name]["measure"]["GP_COMM"]
                        - self.measure[name]["measure"]["GA_COMM"]
                    )
                    self.measure[name]["roofline"]["Util_TOT"] = rf_data["Util"][
                        "Total"
                    ]
                    if utilization_opt_stage == False:
                        breakdown["Util"]["Tot"].extend(
                            [
                                self.measure[name]["measure"]["Util_TOT"],
                                self.measure[name]["roofline"]["Util_TOT"],
                                0,
                            ]
                        )
                    else:
                        breakdown["Util"]["Tot"].extend(
                            [
                                self.measure[name]["measure"]["Util_TOT"],
                                rf_data["Util"]["Total"] * tr / t_noele,
                                rf_data["Util"]["Total"] * tr / t_nocomm,
                                self.measure[name]["roofline"]["Util_TOT"],
                                0,
                            ]
                        )
                    self.measure[name]["measure"]["Util_GP"] = (
                        rf_data["Util"]["GNR"]["PRJ"] * tr_gp / tm_gp
                    )
                    self.measure[name]["roofline"]["Util_GP"] = rf_data["Util"]["GNR"][
                        "PRJ"
                    ]
                    breakdown["Util"]["GP"].extend(
                        [
                            self.measure[name]["measure"]["Util_GP"],
                            self.measure[name]["roofline"]["Util_GP"],
                            0,
                        ]
                    )
                    self.measure[name]["measure"]["Util_GA"] = (
                        rf_data["Util"]["GNR"]["ACT"] * tr_ga / tm_ga
                    )
                    self.measure[name]["roofline"]["Util_GA"] = rf_data["Util"]["GNR"][
                        "ACT"
                    ]
                    breakdown["Util"]["GA"].extend(
                        [
                            self.measure[name]["measure"]["Util_GA"],
                            self.measure[name]["roofline"]["Util_GA"],
                            0,
                        ]
                    )

                    breakdown["Speedup"].extend([1, tm / tr, 1])
        f.close()
        if precentage:
            for s in ["PRE", "GP", "GA"]:
                for p in ["Comp", "Comm", "Ele"]:
                    for i in range(len(breakdown[s][p]) // 3):
                        # breakdown[s][p][i] = breakdown[s][p][i] / breakdown["Tot"][i] if breakdown["Tot"][i] != 0 else 0
                        breakdown[s][p][3 * i] = (
                            breakdown[s][p][3 * i] / breakdown["Tot"][3 * i]
                        )
                        breakdown[s][p][3 * i + 1] = (
                            breakdown[s][p][3 * i + 1] / breakdown["Tot"][3 * i]
                        )
        return breakdown

    def Draw_Measurement(
        self,
        file="measure_vs_roofline.csv",
        figsize=[6, 4],
        Latency_QPS="Latency",
        precentage=False,
        communication=True,
    ):
        breakdown = self.Read_Measurement(file, Latency_QPS, precentage, communication)
        ## draw fig
        fontsize = 5
        width = 0.95
        fig = plt.figure(dpi=300, figsize=figsize)
        ax = plt.gca()

        index = np.arange(0, len(breakdown["Name"]))
        hbars1 = ax.bar(
            index,
            breakdown["PRE"]["Comp"],
            width=width,
            align="center",
            label="PRE-gemm",
            color="maroon",
        )
        acc = np.array(breakdown["PRE"]["Comp"])
        hbars2 = ax.bar(
            index,
            breakdown["PRE"]["Comm"],
            width=width,
            align="center",
            label="PRE-comm",
            color="darkgrey",
            bottom=acc,
        )
        acc += np.array(breakdown["PRE"]["Comm"])
        hbars3 = ax.bar(
            index,
            breakdown["PRE"]["Ele"],
            width=width,
            align="center",
            label="PRE-ele",
            color="green",
            bottom=acc,
        )
        acc += np.array(breakdown["PRE"]["Ele"])
        hbars4 = ax.bar(
            index,
            breakdown["GP"]["Comp"],
            width=width,
            align="center",
            label="GP-gemm",
            color="royalblue",
            bottom=acc,
        )
        acc += np.array(breakdown["GP"]["Comp"])
        hbars5 = ax.bar(
            index,
            breakdown["GP"]["Comm"],
            width=width,
            align="center",
            label="GP-comm",
            color="darkgrey",
            bottom=acc,
        )
        acc += np.array(breakdown["GP"]["Comm"])
        hbars6 = ax.bar(
            index,
            breakdown["GP"]["Ele"],
            width=width,
            align="center",
            label="GP-ele",
            color="green",
            bottom=acc,
        )
        acc += np.array(breakdown["GP"]["Ele"])
        hbars7 = ax.bar(
            index,
            breakdown["GA"]["Comp"],
            width=width,
            align="center",
            label="GA-gemm",
            color="darkorange",
            bottom=acc,
        )
        acc += np.array(breakdown["GA"]["Comp"])
        hbars8 = ax.bar(
            index,
            breakdown["GA"]["Comm"],
            width=width,
            align="center",
            label="GA-comm",
            color="darkgrey",
            bottom=acc,
        )
        acc += np.array(breakdown["GA"]["Comm"])
        hbars9 = ax.bar(
            index,
            breakdown["GA"]["Ele"],
            width=width,
            align="center",
            label="GA-ele",
            color="green",
            bottom=acc,
        )

        ax.set_xticks(
            index,
            ["measure", "roofline", ""] * (len(breakdown["Name"]) // 3),
            minor=False,
            rotation=90,
        )
        ax.set_xticks(index + 0.9, breakdown["Name"], minor=True, rotation=90)

        title = "Measure v.s. Roofline, M6-13B-8K, 4xV100"
        title = (
            "Latency (ms/token/query), "
            if Latency_QPS == "Latency"
            else "Throughput (1/qps), " + title
        )
        if precentage:
            title = "Normalized (%): " + title
        plt.title(title, fontsize=fontsize)

        if precentage == False:
            ax.bar_label(
                hbars9,
                ["{:.1f}x".format(x) if x != 1 else "" for x in breakdown["Speedup"]],
                fontsize=fontsize - 1,
                color="b",
                rotation=90,
                padding=2,
            )
        else:
            ax.bar_label(
                hbars9,
                ["{:.1f}x".format(x) if x != 1 else "" for x in breakdown["Speedup"]],
                fontsize=fontsize - 1,
                color="b",
                rotation=90,
                padding=2,
            )
            ax.bar_label(
                hbars4,
                ["{:.1%}".format(x) if x != 0 else "" for x in breakdown["GP"]["Comp"]],
                fontsize=fontsize - 1,
                color="black",
                rotation=90,
                label_type="center",
            )
            # ax.bar_label(hbars6, ["{:.1%}".format(x) if x != 0 else "" for x in breakdown["GA"]["Comp"]], fontsize = fontsize - 1, color = 'b', rotation = 90, label_type = 'center')

        ax.legend(fontsize=fontsize)

        ax.tick_params(axis="x", which="major", labelsize=fontsize, length=0)
        ax.tick_params(
            axis="x", which="minor", labelsize=fontsize + 0.5, length=20, width=0
        )
        ax.tick_params(axis="y", labelsize=fontsize)
        fig.tight_layout()
        plt.show()

    def Draw_Utilization(
        self,
        file="measure_vs_roofline.csv",
        figsize=[6, 4],
        utilization_opt_stage=False,
    ):
        breakdown = self.Read_Measurement(
            file,
            Latency_QPS="Qps",
            precentage=False,
            communication=True,
            utilization_opt_stage=utilization_opt_stage,
        )
        ## draw fig
        fontsize = 5
        width = 0.95
        fig = plt.figure(dpi=300, figsize=figsize)
        ax = plt.gca()

        index = np.arange(0, len(breakdown["Util"]["Tot"]))
        print(breakdown["Util"]["Tot"])
        hbars1 = ax.bar(
            index,
            breakdown["Util"]["Tot"],
            width=width,
            align="center",
            label="Overall Utilization",
            color="royalblue",
        )

        ax.bar_label(
            hbars1,
            ["{:.1%}".format(x) if x != 0 else "" for x in breakdown["Util"]["Tot"]],
            fontsize=fontsize + 1,
            color="k",
            rotation=90,
            label_type="center",
        )

        ax.set_xticks(
            index,
            (
                ["measure", "ideal ele opt", "ideal comm opt", "roofline", ""]
                if utilization_opt_stage
                else ["measure", "roofline", ""]
            )
            * (len(breakdown["Name"]) // 3),
            minor=False,
            rotation=90,
        )
        ax.set_xticks(
            (
                (np.arange(0, len(breakdown["Name"])) * 5 / 3 + 0.5)
                if utilization_opt_stage
                else index + 0.9
            ),
            breakdown["Name"],
            minor=True,
            rotation=90,
        )

        title = "Measure v.s. Roofline, M6-13B-8K, 4xV100"
        plt.title(title, fontsize=fontsize)

        """
        if precentage == False:
            ax.bar_label(hbars6, breakdown["Speedup"], fontsize = fontsize, color = 'b', rotation = 45)
            ax.bar_label(hbars3, ["Util: {:.1%}".format(x) if x != 0 else "" for x in breakdown["Util"]["Tot"]], fontsize = fontsize - 1, color = 'b', rotation = 90, label_type = 'center')
        else:
            ax.bar_label(hbars3, ["{:.1%}".format(x) if x != 0 else "" for x in breakdown["GP"]["Comp"]], fontsize = fontsize - 1, color = 'b', rotation = 90, label_type = 'center')
            ax.bar_label(hbars5, ["{:.1%}".format(x) if x != 0 else "" for x in breakdown["GA"]["Comp"]], fontsize = fontsize - 1, color = 'b', rotation = 90, label_type = 'center')
        """
        ax.legend(fontsize=fontsize)

        ax.tick_params(axis="x", which="major", labelsize=fontsize, length=0)
        ax.tick_params(
            axis="x", which="minor", labelsize=fontsize + 0.5, length=20, width=0
        )
        ax.tick_params(axis="y", labelsize=fontsize)
        fig.tight_layout()
        plt.show()

    def Optimal_Minimal_Serving_System_Evaluation(
        self,
        hardware,
        model,
        ratiop,
        maxtoken,
        sparse,
        max_FPRT=3,
        min_token_per_sec=20,
        output=True,
        first_col=True,
    ):
        (
            Prefill_Max_Batch_Capacity,
            Prefill_Max_Batch_FPRT,
            Prefill_Max_Batch_Token_Per_Sec,
        ) = self.Get_Max_Batch(
            hardware["PRF"],
            model,
            ratiop,
            maxtoken,
            sparse,
            max_FPRT=max_FPRT,
            min_token_per_sec=min_token_per_sec,
            output=False,
        )
        (
            Generation_Max_Batch_Capacity,
            Generation_Max_Batch_FPRT,
            Generation_Max_Batch_Token_Per_Sec,
        ) = self.Get_Max_Batch(
            hardware["GNR"],
            model,
            ratiop,
            maxtoken,
            sparse,
            max_FPRT=max_FPRT,
            min_token_per_sec=min_token_per_sec,
            output=False,
        )
        Prefill_Max_Batch = min(Prefill_Max_Batch_Capacity, Prefill_Max_Batch_FPRT)
        # print ("Prefill_Max_Batch:", Prefill_Max_Batch)
        # Generation_Max_Batch = 64
        Generation_Max_Batch = min(
            Generation_Max_Batch_Capacity, Generation_Max_Batch_Token_Per_Sec
        )
        # print ("Generation_Max_Batch:", Generation_Max_Batch)
        ### Calculate Prefill
        Prefill_all_data = roofline.Compute_Timebreakdown(
            Ratio_P=[ratiop],
            Batchsize=[Prefill_Max_Batch],
            Max_Token=[maxtoken],
            Model=[model],
            Hardware=[hardware["PRF"]],
            Sparse=[sparse],
            Pipeline_Stage=1,
        )
        Prefill_maxtoken_info = Prefill_all_data[hardware["PRF"]][model][ratiop][
            Prefill_Max_Batch
        ][sparse][maxtoken]
        Prefill_time = (
            Prefill_maxtoken_info["Timebreakdown"]["PRF"]
            + Prefill_maxtoken_info["Timebreakdown"]["OFFCARD"]["PRF"]["Total"]
        )
        _, _, _, Capacity, _ = self.models[model].Calculate_Arithmetic(
            ratiop, Prefill_Max_Batch, maxtoken * ratiop, self.sparses[sparse]
        )
        Prefill_Cap = Capacity["Weight"] + Capacity["KVCache"]
        Prefill_Cap_Util = Prefill_Cap / self.hardwares[hard].MM_Capacity
        # print ("Prefill_Cap:", Prefill_Cap)
        # print ("Total_Cap:", self.hardwares[hard].MM_Capacity)
        # print ("Prefill_time:", Prefill_time)
        ### Calculate Generation
        Generation_all_data = roofline.Compute_Timebreakdown(
            Ratio_P=[ratiop],
            Batchsize=[Generation_Max_Batch],
            Max_Token=[maxtoken],
            Model=[model],
            Hardware=[hardware["GNR"]],
            Sparse=[sparse],
            Pipeline_Stage=1,
        )
        Generation_maxtoken_info = Generation_all_data[hardware["GNR"]][model][ratiop][
            Generation_Max_Batch
        ][sparse][maxtoken]
        Generation_time = (
            Generation_maxtoken_info["Timebreakdown"]["Total"]
            - Generation_maxtoken_info["Timebreakdown"]["PRF"]
            - Generation_maxtoken_info["Timebreakdown"]["OFFCARD"]["PRF"]["Total"]
        )
        _, _, _, Capacity, _ = self.models[model].Calculate_Arithmetic(
            ratiop, Generation_Max_Batch, maxtoken, self.sparses[sparse]
        )
        Generation_Cap = (
            Capacity["Weight"] + Capacity["KVCache"]
        )  # [UP TO UPGRADE] Only consider Homo now
        Generation_Cap_Util = Generation_Cap / self.hardwares[hard].MM_Capacity
        # print ("Generation_time:", Generation_time)
        ### Calculate Machine Number
        Prefill_Machine = math.ceil(Generation_Max_Batch / Prefill_Max_Batch)
        Generation_Machine = math.ceil(Generation_time / Prefill_time)
        ### Calculate Utilization
        Prefill_util = Prefill_maxtoken_info["Util"]["PRF"]
        Generation_util = Generation_maxtoken_info["Util"]["GNR"]["Total"]
        ### Calculate Throughput
        Query_per_Sec = Generation_Max_Batch / Prefill_time
        Query_per_Sec_per_Machine = (
            Generation_Max_Batch / Prefill_time / (Prefill_Machine + Generation_Machine)
        )
        ### Calculate Single Machine Data
        # Batch_Single_Machine = min(Max_Batch_Capacity, Max_Batch_FPRT, Max_Batch_Token_Per_Sec)
        # Single_Machine_all_data = roofline.Compute_Timebreakdown(Ratio_P=[ratiop], Batchsize=[Batch_Single_Machine], Max_Token=[maxtoken], Model=[model], Hardware=[hardware], Sparse=[sparse], Pipeline_Stage = 1)
        # Single_Machine_maxtoken_info = Single_Machine_all_data[hardware][model][ratiop][Batch_Single_Machine][sparse][maxtoken]
        # Single_Machine_total_time = Single_Machine_maxtoken_info["Timebreakdown"]["Total"]
        # Query_per_Sec_Single_Machine = Batch_Single_Machine / Single_Machine_total_time
        if output:
            if first_col:
                print("\033[1;31m*******Serving System*******\033[0m")
                print(
                    "\033[1;31mPRF:Prefill  GNR:Generation  H:Hardware  M:Machine_Number  B:Batch  U:Utilization  T:Time Cap:Capacity\033[0m"
                )
                # print("\033[1;31m%-10s %-8s %-10s %-9s %-1s %-12s %-6s %-6s %-6s %-10s %-1s %-12s %-6s %-6s %-6s %-10s %-6s %-1s %-12s %-11s\033[0m"% \
                print(
                    "\033[1;31m%-10s %-8s %-10s %-1s %-12s %-6s %-6s %-6s %-10s %-6s %-1s %-12s %-6s %-6s %-6s %-10s %-6s %-1s %-8s %-6s\033[0m"
                    % (
                        "Model",
                        "Ratio_P",
                        "Max_Token",
                        "|",
                        "PRF_Hard",
                        "PRF_M",
                        "PRF_B",
                        "PRF_U",
                        "PRF_T(s)",
                        "Cap_U",
                        "|",
                        "GNR_Hard",
                        "GNR_M",
                        "GNR_B",
                        "GNR_U",
                        "GNR_T(s)",
                        "Cap_U",
                        "|",
                        "Q/s",
                        "Q/s/M",
                    )
                )  # , "|", "Single_Batch", "Single_QPS"))
            # print("%-10s %-8.3f %-10d %-9s %-1s %-12s %-6d %-6d %-6.3f %-10.3f %-1s %-12s %-6d %-6d %-6.3f %-10.3f %-6.3f %-1s %-12d %-11.3f"% \
            print(
                "%-10s %-8.3f %-10d %-1s %-12s %-6d %-6d %-6.3f %-10.3f %-6.3f %-1s %-12s %-6d %-6d %-6.3f %-10.3f %-6.3f %-1s %-8.3f %-6.3f"
                % (
                    model,
                    ratiop,
                    maxtoken,
                    "|",
                    hardware["PRF"],
                    Prefill_Machine,
                    Prefill_Max_Batch,
                    Prefill_util,
                    Prefill_time,
                    Prefill_Cap_Util,
                    "|",
                    hardware["GNR"],
                    Generation_Machine,
                    Generation_Max_Batch,
                    Generation_util,
                    Generation_time,
                    Generation_Cap_Util,
                    "|",
                    Query_per_Sec,
                    Query_per_Sec_per_Machine,
                )
            )  # , "|", Batch_Single_Machine, Query_per_Sec_Single_Machine))

        return (
            Prefill_Machine,
            Prefill_Max_Batch,
            Prefill_time,
            Generation_Machine,
            Generation_Max_Batch,
            Generation_time,
        )

    def List_All_Input_Query_Serving_System_Less_Than_Sweet_Point(
        self,
        hardware,
        model,
        ratiop,
        maxtoken,
        sparse,
        max_FPRT=3,
        min_token_per_sec=20,
        output=True,
        first_col=True,
    ):
        if output:
            if first_col:
                print(
                    "\033[1;31m*******List_All_Input_Query_Serving_System_Less_Than_Sweet_Point*******\033[0m"
                )
                print(
                    "\033[1;31mModel:%s, Prefill_Hard:%s, Generation_Hard:%s, Ratio_P:%.3f, Max_Token:%d, max_FPRT(s):%.3f, min_token/s:%.3f\033[0m"
                    % (
                        model,
                        hardware["PRF"],
                        hardware["GNR"],
                        ratiop,
                        maxtoken,
                        max_FPRT,
                        min_token_per_sec,
                    )
                )
                print(
                    "\033[1;31mPRF:Prefill  GNR:Generation  H:Hardware  M:Machine_Number  B:Batch  U:Utilization  T:Time Cap:Capacity\033[0m"
                )
                print(
                    "\033[1;31m%-6s %-1s %-6s %-6s %-10s %-1s %-6s %-6s %-10s %-1s %-10s %-8s %-10s %-10s\033[0m"
                    % (
                        "IN_QPS",
                        "|",
                        "PRF_M",
                        "PRF_B",
                        "PRF_T(s)",
                        "|",
                        "GNR_M",
                        "GNR_B",
                        "GNR_T(s)",
                        "|",
                        "Q/S",
                        "Ttl-M",
                        "Q/s/M",
                        "\xa2/k-token",
                    )
                )

        (
            Opt_Prefill_Machine,
            Opt_Prefill_Max_Batch,
            Opt_Prefill_time,
            Opt_Generation_Machine,
            Opt_Generation_Max_Batch,
            Opt_Generation_time,
        ) = self.Optimal_Minimal_Serving_System_Evaluation(
            hardware,
            model,
            ratiop,
            maxtoken,
            sparse,
            max_FPRT=max_FPRT,
            min_token_per_sec=min_token_per_sec,
            output=False,
            first_col=True,
        )
        # print ("Opt_Prefill_Machine:", Opt_Prefill_Machine)
        Opt_QPS = Opt_Prefill_Max_Batch * Opt_Prefill_Machine / Opt_Prefill_time
        self.cost_model.init_hardware(self.hardwares[hardware["PRF"]])
        PRF_Machine_Price = self.cost_model.cal_cost_per_machine()
        self.cost_model.init_hardware(self.hardwares[hardware["GNR"]])
        GNR_Machine_Price = self.cost_model.cal_cost_per_machine()
        Five_Year_to_Sec = 5 * 365 * 24 * 60 * 60

        Config_Table = {}
        QPS = 1
        while 1:
            # for QPS in range(1, math.ceil(Opt_QPS)):
            Prefill_Machine = math.ceil(QPS * Opt_Prefill_time / Opt_Prefill_Max_Batch)
            Total_Batch = Prefill_Machine * Opt_Prefill_Max_Batch
            if Total_Batch > Opt_Generation_Max_Batch:
                break
            Generation_all_data = roofline.Compute_Timebreakdown(
                Ratio_P=[ratiop],
                Batchsize=[Total_Batch],
                Max_Token=[maxtoken],
                Model=[model],
                Hardware=[hardware["GNR"]],
                Sparse=[sparse],
                Pipeline_Stage=1,
            )
            Generation_maxtoken_info = Generation_all_data[hardware["GNR"]][model][
                ratiop
            ][Total_Batch][sparse][maxtoken]
            Generation_time = (
                Generation_maxtoken_info["Timebreakdown"]["Total"]
                - Generation_maxtoken_info["Timebreakdown"]["PRF"]
                - Generation_maxtoken_info["Timebreakdown"]["OFFCARD"]["PRF"]["Total"]
            )
            Generation_Machine = math.ceil(Generation_time / Opt_Prefill_time)
            QPS_per_Machine = (
                Total_Batch / Opt_Prefill_time / (Prefill_Machine + Generation_Machine)
            )
            Query_per_sec = Total_Batch / Opt_Prefill_time
            Total_Machine = Prefill_Machine + Generation_Machine
            Total_Price = (
                PRF_Machine_Price * Prefill_Machine
                + GNR_Machine_Price * Generation_Machine
            )
            Total_Token_Five_Year = Five_Year_to_Sec * QPS * maxtoken
            Price_per_1k_token = 100 * 1000 * Total_Price / Total_Token_Five_Year
            Config_Table[QPS] = {
                "Prefill_Machine": Prefill_Machine,
                "Prefill_Batch": Opt_Prefill_Max_Batch,
                "Prefill_Time": Opt_Prefill_time,
                "Generation_Machine": Generation_Machine,
                "Generation_Batch": Total_Batch,
                "Generation_Time": Generation_time,
            }
            if output:
                print(
                    "%-6d %-1s %-6d %-6d %-10.3f %-1s %-6d %-6d %-10.3f %-1s %-10.3f %-8d %-10.3f %-10.3f"
                    % (
                        QPS,
                        "|",
                        Prefill_Machine,
                        Opt_Prefill_Max_Batch,
                        Opt_Prefill_time,
                        "|",
                        Generation_Machine,
                        Total_Batch,
                        Generation_time,
                        "|",
                        Query_per_sec,
                        Total_Machine,
                        QPS_per_Machine,
                        Price_per_1k_token,
                    )
                )
            QPS += 1

        return Config_Table

    def List_All_Serving_System_Combination_Fixed_Input_Query(
        self,
        hardware,
        model,
        ratiop,
        maxtoken,
        sparse,
        max_FPRT=3,
        min_token_per_sec=20,
        in_qps=100,
        output=True,
        first_col=True,
    ):
        if output:
            if first_col:
                print(
                    "\033[1;31m*******List_All_Serving_System_Combination_Fixed_Input_Query*******\033[0m"
                )
                print(
                    "\033[1;31mModel:%s, Prefill_Hard:%s, Generation_Hard:%s, Ratio_P:%.3f, Max_Token:%d, max_FPRT(s):%.3f, min_token/s:%.3f, input qps:%.3f\033[0m"
                    % (
                        model,
                        hardware["PRF"],
                        hardware["GNR"],
                        ratiop,
                        maxtoken,
                        max_FPRT,
                        min_token_per_sec,
                        in_qps,
                    )
                )
                print(
                    "\033[1;31mPRF:Prefill  GNR:Generation  H:Hardware  M:Machine_Number  B:Batch  U:Utilization  T:Time Cap:Capacity\033[0m"
                )
                print(
                    "\033[1;31m%-6s %-1s %-6s %-6s %-6s %-1s %-6s %-6s %-6s %-1s %-6s %-6s %-10s\033[0m"
                    % (
                        "IN_QPS",
                        "|",
                        "Rs1",
                        "Rs1_M",
                        "Rs1_B",
                        "|",
                        "Rs2",
                        "Rs2_M",
                        "Rs2_B",
                        "|",
                        "Ttl-M",
                        "Avg_B",
                        "Q/s/M",
                    )
                )
        Config_Table = self.List_All_Input_Query_Serving_System_Less_Than_Sweet_Point(
            hardware,
            model,
            ratiop,
            maxtoken,
            sparse,
            max_FPRT=max_FPRT,
            min_token_per_sec=min_token_per_sec,
            output=False,
            first_col=True,
        )
        max_qps_per_subsystem = max(Config_Table.keys())
        if in_qps <= max_qps_per_subsystem or in_qps >= max_qps_per_subsystem * 2:
            print(
                "in_qps not in (max_qps_per_subsystem, 2*max_qps_per_subsystem)! where in_qps=%.2f, max_qps_per_subsystem=%.2f"
                % (in_qps, max_qps_per_subsystem)
            )
            return
        Rs1 = max_qps_per_subsystem
        Rs2 = in_qps - Rs1
        Rs1_Table = {}
        All_QPS_per_machine = []

        # print (Config_Table[50])
        while Rs1 > 0 and Rs2 <= max_qps_per_subsystem:
            # print (Config_Table[Rs1])
            Machine_Number_Rs1 = (
                Config_Table[Rs1]["Prefill_Machine"]
                + Config_Table[Rs1]["Generation_Machine"]
            )
            # print (Config_Table[Rs2])
            Machine_Number_Rs2 = (
                Config_Table[Rs2]["Prefill_Machine"]
                + Config_Table[Rs2]["Generation_Machine"]
            )
            Total_Machine = Machine_Number_Rs1 + Machine_Number_Rs2
            Query_per_sec_per_machine = in_qps / Total_Machine
            All_QPS_per_machine.append(Query_per_sec_per_machine)
            Generation_Batch_Rs1 = Config_Table[Rs1]["Generation_Batch"]
            Generation_Batch_Rs2 = Config_Table[Rs2]["Generation_Batch"]
            Average_Batch = (
                Generation_Batch_Rs1 * Rs1 + Generation_Batch_Rs2 * Rs2
            ) / in_qps
            Rs1_Table[Rs1] = {
                "Machine_Number_Rs1": Machine_Number_Rs1,
                "Machine_Number_Rs2": Machine_Number_Rs2,
                "Prefill_Machine_Rs1": Config_Table[Rs1]["Prefill_Machine"],
                "Generation_Machine_Rs1": Config_Table[Rs1]["Generation_Machine"],
                "Total_Machine": Total_Machine,
                "Average_Batch": Average_Batch,
                "Query_per_sec_per_machine": Query_per_sec_per_machine,
            }
            if output:
                print(
                    "%-6d %-1s %-6d %-6d %-6d %-1s %-6d %-6d %-6d %-1s %-6d %-6.1f %-10.3f"
                    % (
                        in_qps,
                        "|",
                        Rs1,
                        Machine_Number_Rs1,
                        Generation_Batch_Rs1,
                        "|",
                        Rs2,
                        Machine_Number_Rs2,
                        Generation_Batch_Rs2,
                        "|",
                        Total_Machine,
                        Average_Batch,
                        Query_per_sec_per_machine,
                    )
                )
            Rs1 -= 1
            Rs2 = in_qps - Rs1
        # Max_QPS_per_machine = max(All_QPS_per_machine)
        # Rs1_Table["Max_QPS_per_machine"]=Max_QPS_per_machine
        return Rs1_Table

    def List_All_Serving_System_Range_Input_Query(
        self,
        hardware,
        model,
        ratiop,
        maxtoken,
        sparse,
        max_FPRT=3,
        min_token_per_sec=20,
        output=True,
        first_col=True,
    ):
        if output and first_col:
            print(
                "\033[1;31m*******List_All_Serving_System_Range_Input_Query*******\033[0m"
            )
            print(
                "\033[1;31mModel:%s, Prefill_Hard:%s, Generation_Hard:%s, Ratio_P:%.3f, Max_Token:%d, max_FPRT(s):%.3f, min_token/s:%.3f\033[0m"
                % (
                    model,
                    hardware["PRF"],
                    hardware["GNR"],
                    ratiop,
                    maxtoken,
                    max_FPRT,
                    min_token_per_sec,
                )
            )
            print(
                "\033[1;31mPRF:Prefill  GNR:Generation  H:Hardware  M:Machine_Number  B:Batch  U:Utilization  T:Time Cap:Capacity\033[0m"
            )
            print(
                "\033[1;31m%-6s %-8s %-10s %-10s \033[0m"
                % ("IN_QPS", "Opt_B", "Opt_Q/s/M", "\xa2/k-token")
            )
        Config_Table = self.List_All_Input_Query_Serving_System_Less_Than_Sweet_Point(
            hardware,
            model,
            ratiop,
            maxtoken,
            sparse,
            max_FPRT=max_FPRT,
            min_token_per_sec=min_token_per_sec,
            output=False,
            first_col=True,
        )
        max_qps_per_subsystem = max(Config_Table.keys())
        print(max_qps_per_subsystem)

        self.cost_model.init_hardware(self.hardwares[hardware["PRF"]])
        PRF_Machine_Price = self.cost_model.cal_cost_per_machine()
        self.cost_model.init_hardware(self.hardwares[hardware["GNR"]])
        GNR_Machine_Price = self.cost_model.cal_cost_per_machine()
        Five_Year_to_Sec = 5 * 365 * 24 * 60 * 60

        ALL_QPS = []
        ALL_QPS_per_machine = []
        ALL_Price_per_1k_token = []
        for qps in range(1, max_qps_per_subsystem + 1):
            Prefill_Machine = Config_Table[qps]["Prefill_Machine"]
            Generation_Machine = Config_Table[qps]["Generation_Machine"]
            Total_Price = (
                PRF_Machine_Price * Prefill_Machine
                + GNR_Machine_Price * Generation_Machine
            )
            Total_Token_Five_Year = Five_Year_to_Sec * qps * maxtoken
            Price_per_1k_token = 100 * 1000 * Total_Price / Total_Token_Five_Year
            Max_QPS_per_machine = qps / (Prefill_Machine + Generation_Machine)
            Generation_Batch = Config_Table[qps]["Generation_Batch"]
            ALL_QPS.append(qps)
            ALL_QPS_per_machine.append(Max_QPS_per_machine)
            ALL_Price_per_1k_token.append(Price_per_1k_token)
            if output:
                print(
                    "%-6d %-8.3f %-10.3f %-10.3f"
                    % (qps, Generation_Batch, Max_QPS_per_machine, Price_per_1k_token)
                )
        for qps in range(max_qps_per_subsystem + 1, 2 * max_qps_per_subsystem):
            Rs1_Table = self.List_All_Serving_System_Combination_Fixed_Input_Query(
                hardware,
                model,
                ratiop,
                maxtoken,
                sparse,
                in_qps=qps,
                max_FPRT=max_FPRT,
                min_token_per_sec=min_token_per_sec,
                output=False,
                first_col=True,
            )
            if math.floor(qps / 2) not in Rs1_Table.keys():
                continue
            Prefill_Machine = Rs1_Table[math.floor(qps / 2)]["Prefill_Machine_Rs1"]
            Generation_Machine = Rs1_Table[math.floor(qps / 2)][
                "Generation_Machine_Rs1"
            ]
            Total_Price = (
                PRF_Machine_Price * Prefill_Machine
                + GNR_Machine_Price * Generation_Machine
            ) * 2
            Total_Token_Five_Year = Five_Year_to_Sec * qps * maxtoken
            Price_per_1k_token = 100 * 1000 * Total_Price / Total_Token_Five_Year
            Max_QPS_per_machine = Rs1_Table[math.floor(qps / 2)][
                "Query_per_sec_per_machine"
            ]
            Generation_Batch = Rs1_Table[math.floor(qps / 2)]["Average_Batch"]
            ALL_QPS.append(qps)
            ALL_QPS_per_machine.append(Max_QPS_per_machine)
            ALL_Price_per_1k_token.append(Price_per_1k_token)
            if output:
                print(
                    "%-6d %-8.3f %-10.3f %-10.3f"
                    % (qps, Generation_Batch, Max_QPS_per_machine, Price_per_1k_token)
                )
        return ALL_QPS, ALL_QPS_per_machine, ALL_Price_per_1k_token

    def Evaluate_CPKT_Given_Subsystem(
        self,
        hardware,
        model,
        ratiop,
        maxtoken,
        sparse,
        max_FPRT=3,
        min_token_per_sec=20,
        output=True,
        first_col=True,
    ):
        # Calculate optimal&minimal system and Q_IN_SP
        # For each qps in (0, 3*Q_IN_SP):
        #
        pass

    def MOE_Analysis(self, Ne, Na, Nc, P_List, Batch):
        Max_Expert_Num_List = []
        Prob_List = []
        Active_Comb = []
        for i in range(Na):
            Active_Comb.append(i)
        while 1:
            # Process
            print(Active_Comb)
            if Active_Comb == [Ne - Na + i for i in range(Na)]:
                break
            Active_Comb[-1] += 1
            for i in range(Na - 2, -1, -1):
                if Active_Comb[i + 1] == Ne - Na + i + 2:
                    Active_Comb[i] += 1
                    for j in range(i + 1, Na):
                        Active_Comb[j] = Active_Comb[j - 1] + 1

        return Max_Expert_Num_List, Prob_List

    def MOE_MC(
        self, Total_Expert, Active_Expert, Card_Number, P_List, Batch, Sample=1e5
    ):
        Expert_Per_Card = int(Total_Expert / Card_Number)
        Min_Max = int(Active_Expert * Batch / Card_Number)
        # print (Min_Max)
        Max_Max = min(Active_Expert, Expert_Per_Card) * Batch
        # print (Max_Max)
        Freq = {}
        for i in range(Min_Max, Max_Max + 1):
            Freq[i] = 0
        # print (Freq)
        for s in range(Sample):
            moe_array = np.zeros([Batch, Total_Expert])
            # print (moe_array)
            for row in range(Batch):
                for act in range(Active_Expert):
                    col = random.randint(0, Total_Expert - 1)
                    while moe_array[row, col] == 1:
                        col = random.randint(0, Total_Expert - 1)
                    moe_array[row, col] = 1
            Expert_Vec = np.sum(moe_array, 0)
            # print (Expert_Vec)
            Card_Vec = np.zeros([Card_Number])
            for card in range(Card_Number):
                Card_Vec[card] = np.sum(
                    Expert_Vec[Expert_Per_Card * card : Expert_Per_Card * (card + 1)]
                )
            # print (Card_Vec)
            Max_Expert = max(Card_Vec)
            Freq[Max_Expert] += 1
        print(Freq)
        Average_Max = 0
        for i in range(Min_Max, Max_Max + 1):
            Average_Max += i * Freq[i] / Sample
        print(Average_Max)
        return Average_Max


if __name__ == "__main__":
    roofline = TransformerRoofline(
        "hardware_models.json", "allreduce_v100.xlsx", "hardware_elements.json"
    )
    ### Define New Hete Hardwares
    new_hard = {
        "Name": "8A100+8G6AiM",
        "PRF": {"hard": "A100", "quant": 8},
        "GNR-PRJ": {"hard": "A100", "quant": 8},
        "GNR-ACT": {"hard": "G6AiM", "quant": 8},
        "Assign": "PRF+GNR-PRJ, GNR-ACT",
    }
    roofline.Add_Comb_Hardware(new_hard)
    new_hard = {
        "Name": "8A100+8G6AiM-1",
        "PRF": {"hard": "A100", "quant": 8},
        "GNR-PRJ": {"hard": "G6AiM", "quant": 8},
        "GNR-ACT": {"hard": "G6AiM", "quant": 8},
        "Assign": "PRF, GNR-PRJ+GNR-ACT",
    }
    roofline.Add_Comb_Hardware(new_hard)
    new_hard = {
        "Name": "8A100",
        "PRF": {"hard": "A100", "quant": 8},
        "GNR-PRJ": {"hard": "A100", "quant": 8},
        "GNR-ACT": {"hard": "A100", "quant": 8},
        "Assign": "PRF+GNR-PRJ+GNR-ACT",
        "Type": "Homo",
    }
    roofline.Add_Comb_Hardware(new_hard)
    new_hard = {
        "Name": "8PPU",
        "PRF": {"hard": "PPU", "quant": 8},
        "GNR-PRJ": {"hard": "PPU", "quant": 8},
        "GNR-ACT": {"hard": "PPU", "quant": 8},
        "Assign": "PRF+GNR-PRJ+GNR-ACT",
    }
    roofline.Add_Comb_Hardware(new_hard)
    new_hard = {
        "Name": "8H100",
        "PRF": {"hard": "H100", "quant": 8},
        "GNR-PRJ": {"hard": "H100", "quant": 8},
        "GNR-ACT": {"hard": "H100", "quant": 8},
        "Assign": "PRF+GNR-PRJ+GNR-ACT",
    }
    roofline.Add_Comb_Hardware(new_hard)
    new_hard = {
        "Name": "4V100",
        "PRF": {"hard": "V100", "quant": 4},
        "GNR-PRJ": {"hard": "V100", "quant": 4},
        "GNR-ACT": {"hard": "V100", "quant": 4},
        "Assign": "PRF+GNR-PRJ+GNR-ACT",
        "Type": "Homo",
    }
    roofline.Add_Comb_Hardware(new_hard)
    new_hard = {
        "Name": "8V100",
        "PRF": {"hard": "V100", "quant": 8},
        "GNR-PRJ": {"hard": "V100", "quant": 8},
        "GNR-ACT": {"hard": "V100", "quant": 8},
        "Assign": "PRF+GNR-PRJ+GNR-ACT",
        "Type": "Homo",
    }
    roofline.Add_Comb_Hardware(new_hard)
    new_hard = {
        "Name": "4A100",
        "PRF": {"hard": "A100", "quant": 4},
        "GNR-PRJ": {"hard": "A100", "quant": 4},
        "GNR-ACT": {"hard": "A100", "quant": 4},
        "Assign": "PRF+GNR-PRJ+GNR-ACT",
        "Type": "Homo",
    }
    roofline.Add_Comb_Hardware(new_hard)
    new_hard = {
        "Name": "4THRIVE++",
        "PRF": {"hard": "THRIVE++", "quant": 4},
        "GNR-PRJ": {"hard": "THRIVE++", "quant": 4},
        "GNR-ACT": {"hard": "THRIVE++", "quant": 4},
        "Assign": "PRF+GNR-PRJ+GNR-ACT",
        "Type": "Homo",
    }
    roofline.Add_Comb_Hardware(new_hard)
    new_hard = {
        "Name": "4A10",
        "PRF": {"hard": "A10", "quant": 4},
        "GNR-PRJ": {"hard": "A10", "quant": 4},
        "GNR-ACT": {"hard": "A10", "quant": 4},
        "Assign": "PRF+GNR-PRJ+GNR-ACT",
        "Type": "Homo",
    }
    roofline.Add_Comb_Hardware(new_hard)
    new_hard = {
        "Name": "4H100",
        "PRF": {"hard": "H100", "quant": 4},
        "GNR-PRJ": {"hard": "H100", "quant": 4},
        "GNR-ACT": {"hard": "H100", "quant": 4},
        "Assign": "PRF+GNR-PRJ+GNR-ACT",
        "Type": "Homo",
    }
    roofline.Add_Comb_Hardware(new_hard)
    new_hard = {
        "Name": "120A100",
        "PRF": {"hard": "H100", "quant": 120},
        "GNR-PRJ": {"hard": "H100", "quant": 120},
        "GNR-ACT": {"hard": "H100", "quant": 120},
        "Assign": "PRF+GNR-PRJ+GNR-ACT",
        "Type": "Homo",
    }
    roofline.Add_Comb_Hardware(new_hard)
    new_hard = {
        "Name": "1A100",
        "PRF": {"hard": "A100", "quant": 1},
        "GNR-PRJ": {"hard": "A100", "quant": 1},
        "GNR-ACT": {"hard": "A100", "quant": 1},
        "Assign": "PRF+GNR-PRJ+GNR-ACT",
        "Type": "Homo",
    }
    roofline.Add_Comb_Hardware(new_hard)

    roofline.List_All_Hardware()

    ### Find Max Batch
    max_batch = roofline.Get_Max_Batch_by_Capacity(
        hardware="8A100", model="GPT3", ratiop=0.25, maxtoken=4096, sparse="default"
    )
    if type(max_batch) != int:
        raise ValueError(max_batch)
    else:
        print(
            "For hard:%s, model:%s, ratiop:%.2f, maxtoken:%d, Supported max batch is %d"
            % ("8A100", "GPT3", 0.25, 4096, max_batch)
        )
    max_batch = roofline.Get_Max_Batch_by_Capacity(
        hardware="8A100+8G6AiM",
        model="GPT3",
        ratiop=0.25,
        maxtoken=4096,
        sparse="default",
    )
    if type(max_batch) != int:
        raise ValueError(max_batch)
    else:
        print(
            "For hard:%s, model:%s, ratiop:%.2f, maxtoken:%d, Supported max batch is %d"
            % ("8A100+8G6AiM", "GPT3", 0.25, 4096, max_batch)
        )

    # Model_List = ["M6-10B", "LLaMa", "GPT3.5", "GPT4-8K"]
    Model_List = ["LLaMa2-70B", "LLaMa2-70B-GQA", "LLaMa2-70B-MQA"]
    # Hardware_List = ["4V100", "4A100", "4THRIVE++"]
    Hardware_List = ["8A100"]
    Comb_Hardware_List = [
        {"PRF": "4A100", "GNR": "4A100"},
        {"PRF": "4V100", "GNR": "4V100"},
        {"PRF": "4THRIVE++", "GNR": "4THRIVE++"},
    ]
    # Batchsize_List = [1, 4, 16, 64, 256]
    Batchsize_List = [32]
    # MaxToken_List = [4096]
    MaxToken_List = [4096, 8192, 16384, 32768, 65536]
    RatioP_List = [0.5]
    # RatioP_List = [0.1, 0.25, 0.5, 0.75, 0.9]
    first_col = True
    for model in Model_List:
        for hard in Hardware_List:
            for batch in Batchsize_List:
                for maxtoken in MaxToken_List:
                    for ratiop in RatioP_List:
                        # roofline.List_Timebreakdown_Ratio(Ratio_P=[0.5], Batchsize=[batch], Max_Token=[maxtoken], Model=[model], Hardware=[hard], Sparse=["default"], Pipeline_Stage = 1, first_col=first_col)
                        # roofline.List_Timebreakdown_Absolute(Ratio_P=[ratiop], Batchsize=[batch], Max_Token=[maxtoken], Model=[model], Hardware=[hard], Sparse=["default"], Pipeline_Stage = 1, first_col=first_col)
                        # roofline.Get_Max_Batch(hard, model, 0.5, maxtoken, "default", first_col=first_col)
                        # roofline.Optimal_Minimal_Serving_System_Evaluation({"PRF":hard, "GNR":hard},model,0.5,8192,"default",max_FPRT=1, output=True, first_col=first_col)
                        first_col = False

    hard = "4A100"
    model = "M6-50B-MQ"
    # roofline.List_All_Input_Query_Serving_System_Less_Than_Sweet_Point({"PRF":hard, "GNR":hard},model,0.5,8192,"default", output=True, first_col=True)

    # roofline.List_All_Serving_System_Combination_Fixed_Input_Query({"PRF":hard, "GNR":hard},model,0.5,8192,"default", in_qps=100, output=True, first_col=True)

    Model_List = ["LLaMa2-70B", "LLaMa2-70B-GQA", "LLaMa2-70B-MQA"]
    # Model_List = ["M6-50B-MQ"]
    # Hardware_List = ["4V100", "4A100", "4THRIVE++", "4H100"]
    Hardware_List = ["8A100"]
    # MaxToken_List = [1024, 2048, 4096, 8192, 16384, 32768]
    MaxToken_List = [4096]
    # RatioP_List = [0.01, 0.25, 0.5, 0.75, 0.99]
    RatioP_List = [0.5]
    # FPRT_List = [1,2,3,4,5]
    FPRT_List = [3]
    TokenSpeed_List = [20]
    # TokenSpeed_List = [20]
    plt.figure()
    plt.title("QPS/Machine v.s. RIN")
    plt.xlabel("RIN")
    plt.ylabel("QPS/Machine")
    # plt.title("Cost v.s. RIN")
    # plt.xlabel("RIN")
    # plt.ylabel("Cent/1k-token")
    for ts in TokenSpeed_List:
        for fprt in FPRT_List:
            for ratiop in RatioP_List:
                for maxtoken in MaxToken_List:
                    for model in Model_List:
                        for hard in Hardware_List:
                            # ALL_QPS, ALL_QPS_per_machine, ALL_Price_per_1k_token= roofline.List_All_Serving_System_Range_Input_Query({"PRF":hard, "GNR":hard},model,ratiop,maxtoken,"default", max_FPRT=fprt, min_token_per_sec=ts, output=True, first_col=True)
                            # plt.plot(ALL_QPS, ALL_QPS_per_machine, label=str(model))
                            pass
    plt.legend()
    plt.yscale("log")
    # plt.show()

    first_col = True
    ratiop = 0.5
    for model in Model_List:
        for hard in Hardware_List:
            for batch in Batchsize_List:
                for maxtoken in MaxToken_List:
                    # roofline.List_Timebreakdown_Ratio(Ratio_P=[0.5], Batchsize=[batch], Max_Token=[maxtoken], Model=[model], Hardware=[hard], Sparse=["default"], Pipeline_Stage = 1, first_col=first_col)
                    Max_Batch_Capacity, Max_Batch_FPRT, Max_Batch_Token_Per_Sec = (
                        roofline.Get_Max_Batch(
                            hard,
                            model,
                            ratiop,
                            maxtoken,
                            "default",
                            first_col=first_col,
                            output=False,
                        )
                    )
                    Max_Batch = min(
                        Max_Batch_Capacity, Max_Batch_FPRT, Max_Batch_Token_Per_Sec
                    )
                    all_data = roofline.Compute_Timebreakdown(
                        Ratio_P=[ratiop],
                        Batchsize=[Max_Batch],
                        Max_Token=[maxtoken],
                        Model=[model],
                        Hardware=[hard],
                        Sparse=["default"],
                        Pipeline_Stage=1,
                    )
                    maxtoken_info = all_data[hard][model][ratiop][Max_Batch]["default"][
                        maxtoken
                    ]
                    QPS = Max_Batch / maxtoken_info["Timebreakdown"]["Total"]
                    if first_col:
                        print(
                            "\033[1;31m*******Single Machine Throughput*******\033[0m"
                        )
                        print(
                            "\033[1;31m%-12s %-8s %-10s %-9s %-12s %-10s %-6s\033[0m"
                            % (
                                "Model",
                                "Ratio_P",
                                "Max_Token",
                                "Sparse",
                                "Hardware",
                                "Max_Batch",
                                "Q/s/M",
                            )
                        )
                    print(
                        "%-12s %-8.3f %-10s %-9s %-12s %-10d %-6.3f"
                        % (model, ratiop, maxtoken, "default", hard, Max_Batch, QPS)
                    )
                    first_col = False

    # roofline.MOE_Analysis(32, 8, 0, 0, 0)
    # roofline.MOE_MC(Total_Expert=8, Active_Expert=4, Card_Number=4, P_List=0, Batch=4, Sample=100000)
    roofline.List_Timebreakdown_Absolute(
        Ratio_P=[0.5],
        Batchsize=[8],
        Max_Token=[8192],
        Model=["Transformer-6.7B"],
        Hardware=["1A100"],
        Sparse=["default"],
        Pipeline_Stage=1,
        first_col=False,
    )

    # For Prefill, Step=0
    Prefill_PRJ_Latency, Prefill_ACT_Latency = roofline.Compute_Timebreakdown_Iteration(
        Prompt_Len=1024, Step=0, Batchsize=2, Model="Transformer-6.7B", Hardware="4A100"
    )
    # For Generation, Step>0
    Generation_PRJ_Latency, Generation_ACT_Latency = (
        roofline.Compute_Timebreakdown_Iteration(
            Prompt_Len=1024,
            Step=100,
            Batchsize=2,
            Model="Transformer-6.7B",
            Hardware="4A100",
        )
    )
    print(Prefill_PRJ_Latency)
    print(Prefill_ACT_Latency)
    print(Generation_PRJ_Latency)
    print(Generation_ACT_Latency)

    # trace = roofline.Generate_Trace(Prompt_Len=1000, Generation_Len=2000, Batchsize=256, Model="GPT3.5", Hardware="4A100")
    # for i in trace:
    #     print (i)
    new_hard = {
        "Name": "2A100",
        "PRF": {"hard": "A100", "quant": 2},
        "GNR-PRJ": {"hard": "A100", "quant": 2},
        "GNR-ACT": {"hard": "A100", "quant": 2},
        "Assign": "PRF+GNR-PRJ+GNR-ACT",
        "Type": "Homo",
        "AllReduce_Links": ["NVLink"],
    }
    roofline.Add_Comb_Hardware(new_hard)
    new_hard = {
        "Name": "2A100-DDR",
        "PRF": {"hard": "A100", "quant": 2},
        "GNR-PRJ": {"hard": "A100", "quant": 2},
        "GNR-ACT": {"hard": "A100", "quant": 2},
        "Assign": "PRF+GNR-PRJ+GNR-ACT",
        "Type": "Homo",
        "AllReduce_Links": ["DDR"],
        "DDR_Channel": 2,
    }
    roofline.Add_Comb_Hardware(new_hard)
    new_hard = {
        "Name": "2A100-DDR+RoCE",
        "PRF": {"hard": "A100", "quant": 2},
        "GNR-PRJ": {"hard": "A100", "quant": 2},
        "GNR-ACT": {"hard": "A100", "quant": 2},
        "Assign": "PRF+GNR-PRJ+GNR-ACT",
        "Type": "Homo",
        "AllReduce_Links": ["DDR", "RoCE"],
        "DDR_Channel": 2,
    }
    roofline.Add_Comb_Hardware(new_hard)
    roofline.List_Timebreakdown_Absolute(
        Ratio_P=[0.5],
        Batchsize=[8],
        Max_Token=[8192],
        Model=["Transformer-6.7B"],
        Hardware=["2A100"],
        Sparse=["default"],
        Pipeline_Stage=1,
        first_col=True,
    )
    roofline.List_Timebreakdown_Absolute(
        Ratio_P=[0.5],
        Batchsize=[8],
        Max_Token=[8192],
        Model=["Transformer-6.7B"],
        Hardware=["2A100-DDR"],
        Sparse=["default"],
        Pipeline_Stage=1,
        first_col=False,
    )
    roofline.List_Timebreakdown_Absolute(
        Ratio_P=[0.5],
        Batchsize=[8],
        Max_Token=[8192],
        Model=["GPT3.5"],
        Hardware=["2A100-DDR+RoCE"],
        Sparse=["default"],
        Pipeline_Stage=1,
        first_col=False,
    )
    new_hard = {
        "Name": "THRIVE_Stage2",
        "PRF": {"hard": "THRIVE_LOL", "quant": 32},
        "GNR-PRJ": {"hard": "THRIVE_LOL", "quant": 32},
        "GNR-ACT": {"hard": "THRIVE_LOL", "quant": 32},
        "Assign": "PRF+GNR-PRJ+GNR-ACT",
        "Type": "Homo",
        "AllReduce_Links": ["DDR", "FPGA-8Card"],
        "DDR_Channel": 32,
    }
    roofline.Add_Comb_Hardware(new_hard)
    roofline.List_Timebreakdown_Absolute(
        Ratio_P=[0.5],
        Batchsize=[64],
        Max_Token=[8192],
        Model=["GPT3.5"],
        Hardware=["THRIVE_Stage2"],
        Sparse=["default"],
        Pipeline_Stage=1,
        first_col=True,
    )
    roofline.List_Timebreakdown_Absolute(
        Ratio_P=[0.5],
        Batchsize=[64],
        Max_Token=[8192],
        Model=["GPT3.5"],
        Hardware=["8A100"],
        Sparse=["default"],
        Pipeline_Stage=1,
        first_col=False,
    )
    roofline.List_Timebreakdown_Absolute(
        Ratio_P=[0.5],
        Batchsize=[1, 2, 4, 8, 16, 32, 64],
        Max_Token=[8192],
        Model=["M6-50B"],
        Hardware=["THRIVE_Stage2"],
        Sparse=["default"],
        Pipeline_Stage=1,
        first_col=True,
    )
    DDR_List = ["12.5", "20", "25", "50", "75", "100"]
    FPGA_List = ["12.5", "25", "50", "100", "150"]
    name_List = []
    for ddr in DDR_List:
        for fpga in FPGA_List:
            new_hard = {
                "Name": "DDR-" + ddr + "-FPGA-" + fpga,
                "PRF": {"hard": "THRIVE_LOL", "quant": 32},
                "GNR-PRJ": {"hard": "THRIVE_LOL", "quant": 32},
                "GNR-ACT": {"hard": "THRIVE_LOL", "quant": 32},
                "Assign": "PRF+GNR-PRJ+GNR-ACT",
                "Type": "Homo",
                "AllReduce_Links": ["DDR-" + ddr, "FPGA-8Card-" + fpga],
                "DDR_Channel": 32,
            }
            name_List.append("DDR-" + ddr + "-FPGA-" + fpga)
            roofline.Add_Comb_Hardware(new_hard)
    is_first_col = True
    for ddr in DDR_List:
        for fpga in FPGA_List:
            roofline.List_Timebreakdown_Absolute(
                Ratio_P=[0.5],
                Batchsize=[64],
                Max_Token=[8192],
                Model=["M6-50B"],
                Hardware=["DDR-" + ddr + "-FPGA-" + fpga],
                Sparse=["default"],
                Pipeline_Stage=1,
                first_col=is_first_col,
            )
            is_first_col = False
        print("")
    new_hard = {
        "Name": "1A100-40G",
        "PRF": {"hard": "A100-40G", "quant": 1},
        "GNR-PRJ": {"hard": "A100-40G", "quant": 1},
        "GNR-ACT": {"hard": "A100-40G", "quant": 1},
        "Assign": "PRF+GNR-PRJ+GNR-ACT",
        "Type": "Homo",
    }
    roofline.Add_Comb_Hardware(new_hard)
    roofline.List_Timebreakdown_Absolute(
        Ratio_P=[0.125],
        Batchsize=[1],
        Max_Token=[8192],
        Model=["M6-50B"],
        Hardware=["1A100-40G"],
        Sparse=["default"],
        Pipeline_Stage=1,
        first_col=is_first_col,
    )
    # roofline.List_Timebreakdown_Absolute(Ratio_P=[0.25], Batchsize=[1, 2, 4, 8, 16, 32, 64], Max_Token=[8192], Model=["M6-50B"], Hardware=["8A100"], Sparse=["default"], Pipeline_Stage = 1, first_col=True)
    # roofline.List_Timebreakdown_Absolute(Ratio_P=[0.5], Batchsize=[1], Max_Token=[8192], Model=["M6-50B"], Hardware=["8A100"], Sparse=["default"], Pipeline_Stage = 1, first_col=False)
    # timebreakdown = roofline.Compute_Timebreakdown(Ratio_P=[0.5], Batchsize=[1], Max_Token=[8192], Model=["GPT3.5"], Hardware=["THRIVE_Stage2"], Sparse=["default"], Pipeline_Stage = 1, first_col=False)
    # print(timebreakdown[])
    # prefill_time = []
    # for Prompt_Len in range(1, 1025):
    #     PRJ_Time, ACT_Time = roofline.Compute_Timebreakdown_Iteration(Prompt_Len, 0, 1, "OPT-13B", "A100-40G", Sparse="default", Pipeline_Stage = 1)
    #     # print (PRJ_Time + ACT_Time, PRJ_Time, ACT_Time)
    #     prefill_time.append(PRJ_Time + ACT_Time)
    # for Prompt_Len in range(1, 1024):
    #     Target_Len = int(Prompt_Len / 128)*128 + 128
    #     print (prefill_time[Target_Len-1]*1.41+0.009)

    # roofline.List_Timebreakdown_Absolute(Ratio_P=[0.9999], Batchsize=[1], Max_Token=[32768], Model=["GPT3.5"], Hardware=["8V100", "8A100"], Sparse=["default"], Pipeline_Stage = 1, first_col=True)
