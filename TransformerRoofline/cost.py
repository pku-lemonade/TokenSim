import json
import matplotlib.pyplot as plt
import numpy as np
import copy
# from roofline import TransformerRoofline

# XPU Card
XPU = ["A100", "V100", "A10", "A100-DGX", "THRIVE2", "THRIVE4"]

# PIM Card
# G6AiM Card = 16 G6-chips
# HBM-PIM Card = 5 HBM-PIM chips
# Upmem Card = 20 Upmem DIMMs
# LPDDR4-PIM-DIMM Card = 32 LPDDR4-PIM Chip (small)
# LPDDR4-PIM-PCIe Card = 1 LPDDR4-PIM Chip (large)
PIM = ["G6AiM", "HBM-PIM", "Upmem", "LPDDR4-PIM-DIMM", "LPDDR4-PIM-PCIe", "PIM-DSE"]

class Hardware_Element:
    def __init__(self, input_hardware):
        self.Name = input_hardware["Name"]
        self.Form_Type = input_hardware["Type"]
        self.Static_Power = input_hardware["Static_Power"]
        self.TDP = input_hardware["TDP"]
        self.num = None
        self.price = input_hardware["Price"] if "Price" in input_hardware else None
        self.util = 1.0
        self.TFLOPS = input_hardware["TFLOPS"] if "TFLOPS" in input_hardware else None
        self.BW_TBs = input_hardware["BW_TBs"] if "BW_TBs" in input_hardware else None
        self.Capacity = input_hardware["Capacity"] if "Capacity" in input_hardware else None
        self.Memory_Tech_Node = input_hardware["Memory_Tech_Node"] if "Memory_Tech_Node" in input_hardware else None
        self.Logic_Tech_Node = input_hardware["Logic_Tech_Node"] if "Logic_Tech_Node" in input_hardware else None
        self.Memory_Layer = input_hardware["Memory_Layer"] if "Memory_Layer" in input_hardware else None
        self.Die_Area = input_hardware["Die_Area"] if "Die_Area" in input_hardware else None

    def __str__(self):
        f = "name: {}\nnum: {}\ntype: {}\nstatic_power: {}\nTDP: {}\nutil: {}\nprice: {}\nTFLOPS: {}\nBW_TBs: {}\nCapacity: {}\nMemory_Tech_Node: {}\nLogic_Tech_Node: {}\nMemory_Layer: {}\nDie_Area: {}\n"
        return f.format(self.Name, self.num, self.Form_Type, self.Static_Power, self.TDP, self.util, self.price, self.TFLOPS, self.BW_TBs, self.Capacity, self.Memory_Tech_Node, self.Logic_Tech_Node, self.Memory_Layer, self.Die_Area)
        
class CostEst:
    def __init__(self, hardware_json):
        self.hardware_elements = {}
        with open(hardware_json, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            #for hardware in data["hardware"]:
            #    name = hardware["Name"]
            #    self.hardware_elements[name] = Hardware_Element(hardware)
            self.server_price = data["server_price"]
            self.avg_server_power = data["avg_server_power"]
            self.electric_price = data["electric_price"]
            self.design_cost = data["design_cost"]
            self.wafer_fee_dram = data["wafer_fee_dram"]
            self.wafer_fee_logic = data["wafer_fee_logic"]
            self.time_yrs = data["time_yrs"]
            self.required_token_per_second = data["Require_Token_per_Second"]
                
    def init_hardware(self, hardware_config):
        """
        hardware_config: hardware configuration
        """
        self.config = ""
        self.XPU = None
        self.PIM = None
        for hw in hardware_config:
            if hw in XPU:
                self.XPU = self.hardware_elements[hw]
                self.XPU.num = hardware_config[hw]
                self.config = self.config + hw + "x" + str(self.XPU.num) + " + "
            elif hw in PIM:
                self.PIM = self.hardware_elements[hw]
                self.PIM.num = hardware_config[hw]
                self.config = self.config + hw + "x" + str(self.PIM.num)

        self.server_num = hardware_config["Server"]
        self.rack_unit = hardware_config["Rack_Unit"]
        self.required_token_per_second = hardware_config["Require_Token_per_Second"]
        self.token_per_second = hardware_config["Token_per_Second"]

    def init_hardware(self, comb_hw, token_per_second = 0):
        """
        hardware_config: hardware configuration
        """
        self.config = comb_hw.Name
        self.XPU = copy.deepcopy(comb_hw.XPU)
        self.XPU.num = comb_hw.MM_Card_Num
        self.PIM = copy.deepcopy(comb_hw.PIM)
        if self.PIM:
            self.PIM.num = comb_hw.MV_Card_Num
            self.PIM_capacity = comb_hw.MV_Capacity
        self.pim_layer = comb_hw.pim_layer

        self.server_num = comb_hw.server_num
        self.rack_unit = comb_hw.rack_unit
        self.assign = comb_hw.Assign
        self.type = comb_hw.Form_Type
        #self.required_token_per_second = comb_hw["Require_Token_per_Second"]
        self.token_per_second = token_per_second

    def cal_total_cost(self, token_per_second = 0):
        """
        required_token_per_second: required token per second
        token_per_second: hetero or homo hardware performance, calculated by roofline model
        """
        if token_per_second != 0:
            self.token_per_second = token_per_second
        self.time_op_hrs = self.time_yrs * 365 * 24
        self.time_op_mins = self.time_yrs * 365 * 24 * 60
        self.time_op_secs = self.time_yrs * 365 * 24 * 60 * 60
        
        ## calculate price per device
        self.device_num = self.required_token_per_second / self.token_per_second
        if self.XPU.price == None or self.XPU.price == 0:
            self.XPU.price = self.cal_x_price(self.XPU.Name, self.device_num * self.XPU.num)
        self.XPU_price = self.XPU.price * self.XPU.num
        self.operation_price_XPU = self.XPU.num * (self.XPU.Static_Power+(self.XPU.TDP-self.XPU.Static_Power)*self.XPU.util)/1000 * self.time_op_hrs * self.electric_price
        
        self.PIM_price = 0
        self.operation_price_PIM = 0
        if self.PIM: # if PIM exists
            if self.PIM.price == None or self.PIM.price == 0:
                self.PIM.price = self.cal_x_price(self.PIM.Name, self.device_num * self.PIM.num)
            self.PIM_price = self.PIM.price * self.PIM.num
            self.operation_price_PIM = self.PIM.num * (self.PIM.Static_Power+(self.PIM.TDP-self.PIM.Static_Power)*self.PIM.util)/1000 * self.time_op_hrs * self.electric_price
        
        ## calculate server price
        self.server_total_price = self.server_num * self.server_price
        
        ## calculate operation price
        self.operation_price_server = self.server_num * self.avg_server_power / 1000 * self.time_op_hrs * self.electric_price
        self.operation_price = self.operation_price_XPU + self.operation_price_PIM + self.operation_price_server
        
        ## calculate datacenter operation fee
        self.datacenter_op_fee = 18000 * self.rack_unit / 42 * self.time_yrs
        
        ## calculate device price
        self.device_price = self.XPU_price + self.PIM_price
        
        self.total_cost_per_device = self.device_price + self.server_total_price + self.operation_price + self.datacenter_op_fee
        self.total_cost = self.total_cost_per_device * self.device_num
        self.price_per_token = self.total_cost / (self.required_token_per_second * self.time_op_secs)
        return self.price_per_token
    
    def cal_cost_per_machine(self):
        """
        required_token_per_second: required token per second
        token_per_second: hetero or homo hardware performance, calculated by roofline model
        """
        self.time_op_hrs = self.time_yrs * 365 * 24
        self.time_op_mins = self.time_yrs * 365 * 24 * 60
        self.time_op_secs = self.time_yrs * 365 * 24 * 60 * 60
        
        ## calculate price per device
        self.device_num = 1
        # print("XPU price: {}".format(self.XPU.price)) 
        if self.XPU.price == None or self.XPU.price == 0:
            # print("total num: {}".format(self.device_num * self.XPU.num))
            self.XPU.price = self.cal_x_price(self.XPU.Name, self.device_num * self.XPU.num)
            # print("after caculation, XPU price: {}".format(self.XPU.price))
        self.XPU_price = self.XPU.price * self.XPU.num
        
        self.operation_price_XPU = self.XPU.num * (self.XPU.Static_Power+(self.XPU.TDP-self.XPU.Static_Power)*self.XPU.util)/1000 * self.time_op_hrs * self.electric_price
        
        self.PIM_price = 0
        self.operation_price_PIM = 0
        if self.PIM: # if PIM exists
            # print("PIM price: {}".format(self.PIM.price)) 
            if self.PIM.price == None or self.PIM.price == 0:
                self.PIM.price = self.cal_x_price(self.PIM.Name, self.device_num * self.PIM.num)
                # print("after caculation, PIM price: {}".format(self.PIM.price)) 
            self.PIM_price = self.PIM.price * self.PIM.num
            self.operation_price_PIM = self.PIM.num * (self.PIM.Static_Power+(self.PIM.TDP-self.PIM.Static_Power)*self.PIM.util)/1000 * self.time_op_hrs * self.electric_price
        
        ## calculate server price
        self.server_total_price = self.server_num * self.server_price
        
        ## calculate operation price
        self.operation_price_server = self.server_num * self.avg_server_power / 1000 * self.time_op_hrs * self.electric_price
        self.operation_price = self.operation_price_XPU + self.operation_price_PIM + self.operation_price_server
        
        ## calculate datacenter operation fee
        self.datacenter_op_fee = 18000 * self.rack_unit / 42 * self.time_yrs
        
        ## calculate device price
        self.device_price = self.XPU_price + self.PIM_price
        
        self.total_cost_per_device = self.device_price + self.server_total_price + self.operation_price + self.datacenter_op_fee
        self.total_cost = self.total_cost_per_device * self.device_num
        
        return self.total_cost

    def cal_system_cost(self, token_per_second=0, num_of_PRF_system=0, num_of_GNR_system=0, num_of_mem_layer=1, Assign="PRF+GNR-PRJ+GNR-ACT"):
        """
        input:
        - token_per_second: serving system throughput
        - num_of_PRF_system: number of device for PRF, note the device should be combination hardware like 8*A100
        - num_of_GNR_system: number of device for GNR, note the device should be combination hardware like 8*A100, or 8*A100+m*CXLv0, ...
        - Assign: indicate the system config is heterogenous or homogenous
        Note: Only support same GPU in PRF and GNR; such as PRF: 8*A100, GNR: 8*A100 or PRF: 8*H100, GNR: 8*H100+32*CXLv0
        """
        assert Assign in ["PRF+GNR-PRJ+GNR-ACT", "PRF+GNR-PRJ, GNR-ACT"], " not supportd now!".format(Assign)
        if token_per_second != 0: self.token_per_second = token_per_second

        self.time_op_hrs = self.time_yrs * 365 * 24
        self.time_op_mins = self.time_yrs * 365 * 24 * 60
        self.time_op_secs = self.time_yrs * 365 * 24 * 60 * 60

        # Total device requirement for given required token/s
        self.PRF_device = self.required_token_per_second / self.token_per_second * num_of_PRF_system
        self.GNR_device = self.required_token_per_second / self.token_per_second * num_of_GNR_system

        if Assign == "PRF+GNR-PRJ+GNR-ACT":
            # the PRF and GNR are homogeneous system
            self.XPU_device = self.PRF_device + self.GNR_device
            self.PIM_device = 0
        elif Assign == "PRF+GNR-PRJ, GNR-ACT":
            # the GNR is heterogenous system
            self.XPU_device = self.PRF_device + self.GNR_device 
            self.PIM_device = self.GNR_device 

        # Calculate XPU price
        if self.XPU.price == None or self.XPU.price == 0:
            # No price available, should be calculated from TCO
            assert self.XPU.Name in ["THRIVE++", "PPU"], " Only support XPU==THRIVE++ or XPU==PPU exploration now!"
            self.XPU.price = self.cal_x_price(chip=self.XPU.Name, num=self.XPU_device * self.XPU.num, num_of_mem_layer=num_of_mem_layer)
                
        self.XPU_device_price = self.XPU.price * self.XPU.num
        self.XPU_op_cost = self.XPU.num * (self.XPU.Static_Power+(self.XPU.TDP-self.XPU.Static_Power)*self.XPU.util)/1000 * self.time_op_hrs * self.electric_price

        # Calculate PIM device price
        self.PIM_device_price = 0
        self.PIM_op_cost = 0
        if self.PIM: # if PIM exists
            if self.PIM.price == None or self.PIM.price == 0:
                self.PIM.price = self.cal_x_price(self.PIM.Name, self.PIM_device * self.PIM.num, num_of_mem_layer)
            self.PIM_device_price = self.PIM.price * self.PIM.num
            self.PIM_op_cost = self.PIM.num * (self.PIM.Static_Power+(self.PIM.TDP-self.PIM.Static_Power)*self.PIM.util)/1000 * self.time_op_hrs * self.electric_price
        
        ## calculate server cost
        self.server_cost = self.server_num * self.server_price
        ## calculate server operation cost
        self.server_op_cost = self.server_num * self.avg_server_power / 1000 * self.time_op_hrs * self.electric_price

        # device cost
        self.device_cost = self.XPU_device_price + self.PIM_device_price
        
        # calculate total device cost
        self.total_device_cost = self.XPU_device_price * self.XPU_device + self.PIM_device_price * self.PIM_device
        # calculate service cost; [FIXME] assume PRF device server num is same as GNR
        self.total_server_cost = self.server_cost * (self.PRF_device + self.GNR_device) 
        # calculate total operation cost
        self.total_op_cost = self.XPU_op_cost * self.XPU_device + self.PIM_op_cost * self.PIM_device + self.server_op_cost * (self.PRF_device + self.GNR_device) 
        ## calculate datacenter operation cost; [FIXME] assume PRF device rack unit is same as GNR
        self.datacenter_op_cost = (18000 * self.rack_unit / 42 * self.time_yrs) * (self.PRF_device + self.GNR_device)

        self.total_cost = self.total_device_cost + self.total_server_cost + self.total_op_cost + self.datacenter_op_cost
        self.price_per_token = self.total_cost / (self.required_token_per_second * self.time_op_secs)
        return self.price_per_token

    def cal_x_price(self, chip=None, num=0, num_of_mem_layer=1):
        """
        calculate NRE for new chip design
        """

        if chip == "DIMMv0":
            chip_area = 100
            chip_yield = 0.95
            wafer_cost = self.wafer_fee_dram * 3 * (num*32) / (np.pi * 150 * 150 / chip_area * chip_yield) # 1 DIMM = 32 Chips
        elif chip == "CXLv0":
            # [FIXME] hardcode
            # dram wafer price: $5000, bonding price $1200, logic wafer price: $3000 (28nm)
            wafer_cost = (self.pim_layer * (5000 + 1200) + 3000) * 7.2 * num / 66
            # wafer_cost = self.wafer_fee_dram * 3 * (num*1) / (np.pi * 150 * 150 / chip_area * chip_yield) # 1 M.2/PCIe = 1 Chip
        elif chip == "THRIVE2":
            chip_area = 800
            chip_yield = 0.8
            wafer_cost = (self.wafer_fee_dram * 2 + self.wafer_fee_logic) * num / (np.pi * 150 * 150 / chip_area * chip_yield)
        elif chip == "THRIVE4":
            chip_area = 800
            chip_yield = 0.8
            wafer_cost = (self.wafer_fee_dram * 4 + self.wafer_fee_logic) * num / (np.pi * 150 * 150 / chip_area * chip_yield)
        elif chip == "THRIVE++MCM":
            chip_area = 800
            chip_yield = 0.8
            wafer_cost = (self.wafer_fee_dram * 8 + self.wafer_fee_logic) * (num*2) / (np.pi * 150 * 150 / chip_area * chip_yield)
        # elif chip == "THRIVE++":
        #     chip_area = 800
        #     chip_yield = 0.8
        #     wafer_cost = (self.wafer_fee_dram * 8 + self.wafer_fee_logic) * num / (np.pi * 150 * 150 / chip_area * chip_yield)
        elif chip == "THRIVE++":
            # assume $1 = ￥7.2; "66" is yield data from ICL
            wafer_cost =  (num_of_mem_layer * (5000 + 1200) + 3000) * 7.2 * num / 66 
            # wafer_cost =  (num_of_mem_layer * (5000 + 1200) + 3000*2 + 1200) * 7.2 * num / 66 # used for auxiliary die
            return (wafer_cost + self.design_cost) / num
        elif chip == "PPU":
            chip_area = 600
            chip_yield = 0.92
            # 7nm wafer price: $9346
            wafer_cost = 9346 * 7.2 * num / (np.pi * 150 * 150 / chip_area * chip_yield)
        elif chip == "PIMv1": # [FIXME] PIMv1 cost is related with Capacity (memory area)
            wafer_cost =  (num_of_mem_layer * (5000 + 1200) + 3000) * 7.2 * (num * 32) / 66
            # wafer_cost =  (num_of_mem_layer * (5000 + 1200) + 3000*2 + 1200) * 7.2 * (num * 32) / 66 # used for auxiliary die
            return (wafer_cost + self.design_cost) / num
        return (wafer_cost + self.design_cost) / num

    def List_Price_Breakdown(self, fig=True):
        print(50*"*");print("Computation System Configuration");print(50*"*");
        if self.XPU: print(self.XPU)
        if self.PIM: print(self.PIM)
        print("\n")

        print(50*"*");print("Price Breakdown");print(50*"*")
        print("1. device price: {:.0f}".format(self.device_price))
        print("-> XPU system price: {:.0f}".format(self.XPU_price))
        print("---> XPU card price: {:.0f}".format(self.XPU.price))
        print("-> PIM system price: {:.0f}".format(self.PIM_price))
        if self.PIM: print("---> PIM card price: {:.0f} (note this is card price, not chip price)".format(self.PIM.price))
        print("2. server total price: {:.0f}".format(self.server_total_price))
        print("3. operation price for {} yrs: {:.0f}".format(self.time_yrs, self.operation_price))
        print("-> XPU operation price: {:.0f}".format(self.operation_price_XPU))
        print("-> PIM operation price: {:.0f}".format(self.operation_price_PIM))
        print("-> Server operation price: {:.0f}".format(self.operation_price_server))
        print("4. datacenter price: {:.0f}".format(self.datacenter_op_fee))
        print("total_cost_per_device: {:.0f}".format(self.total_cost_per_device))
        print("\n")

        print(50*"*");print("Computation System Requirement");print(50*"*")
        print("Required Token per Seconde: {:e}".format(self.required_token_per_second))
        print("Required Computation System: {:.0f}".format(self.device_num))
        print("-> XPU card requirements: {:.0f}".format(self.device_num * self.XPU.num))
        if self.PIM:
            print("-> PIM card requirements: {:.0f} (note this is card requirement, not chip)".format(self.device_num * self.PIM.num))
        print("\n")

        print(50*"*");print("Total Cost");print(50*"*")
        print("total_cost (TCO): {:.0f}".format(self.total_cost))
        print("price_per_token: {:.3e}".format(self.price_per_token))

        print("\n")
        print("\033[1;31m Note all the price unit is ￥RMB\033[0m")

        if fig:
            plt.figure(figsize=(6,6))
            price = [
                self.XPU_price, 
                self.PIM_price, 
                self.operation_price_XPU, 
                self.operation_price_PIM, 
                self.operation_price_server, 
                self.server_total_price, 
                self.datacenter_op_fee
            ]
            labels = ["XPU", "PIM", "op_XPU", "op_PIM", "op_Server", "Server", "Others"]
            plt.pie(price, labels=labels, autopct="%1.1f%%", textprops={'fontsize':25})
            plt.title("Price Breakdown: "+self.config, fontdict={'fontsize':25})
            plt.show()
'''
def test_0_0():
    """
    test_0_0: use cost model
    """
    Require_Token_per_Second = 99000000
    cost_config = {"A100":8, "G6AiM":8, "Server": 2, "Rack_Unit": 8,"Require_Token_per_Second": Require_Token_per_Second,"Token_per_Second": 1150.34}
    cost_model = CostEst(hardware_json="hardware_elements.json")
    cost_model.init_hardware(hardware_config=cost_config)
    cost_model.cal_total_cost()
    cost_model.List_Price_Breakdown(fig=True)

def test_0_1():
    """
    test_0_1: use cost model with roofline model
    """
    Require_Token_per_Second = 99000000
    cost_config = {"A100":8, "G6AiM":8, "Server": 2, "Rack_Unit": 8,"Require_Token_per_Second": Require_Token_per_Second,"Token_per_Second": 0}
    roofline_config = [{"Name":"1", "PRF":{"hard":"A100", "quant":8}, "GNR-PRJ":{"hard":"A100", "quant":8}, "GNR-ACT":{"hard":"G6AiM", "quant":8}, "Assign": "PRF+GNR-PRJ, GNR-ACT"}] 
    roofline = TransformerRoofline('hardware_models.json')
    roofline.Draw_Combination_Timebreakdown(Ratio_P=[0.25], Batchsize=[32], Max_Token=[4096], Model=["GPT3.5"], Comb_Hardware=roofline_config, Sparse=["default"])
    cost_config["Token_per_Second"] = roofline.Token_per_Sec[0]
    cost_model = CostEst(hardware_json="hardware_elements.json")
    cost_model.init_hardware(hardware_config=cost_config)
    cost_model.cal_total_cost()
    cost_model.List_Price_Breakdown(fig=True)

def test_0_2():
    """
    test_0_1: use cost model with roofline model, test for A10 and V100
    """
    Require_Token_per_Second = 99000000
    cost_config = {"A10":8, "G6AiM":8, "Server": 2, "Rack_Unit": 8,"Require_Token_per_Second": Require_Token_per_Second,"Token_per_Second": 0}
    roofline_config = [{"Name":"1", "PRF":{"hard":"A10", "quant":8}, "GNR-PRJ":{"hard":"A10", "quant":8}, "GNR-ACT":{"hard":"G6AiM", "quant":8}, "Assign": "PRF+GNR-PRJ, GNR-ACT"}] 
    roofline = TransformerRoofline('hardware_models.json')
    roofline.Draw_Combination_Timebreakdown(Ratio_P=[0.25], Batchsize=[32], Max_Token=[8192], Model=["GPT3.5"], Comb_Hardware=roofline_config, Sparse=["default"])
    cost_config["Token_per_Second"] = roofline.Token_per_Sec[0]
    cost_model = CostEst(hardware_json="hardware_elements.json")
    cost_model.init_hardware(hardware_config=cost_config)
    cost_model.cal_total_cost()
    cost_model.List_Price_Breakdown(fig=True)

    cost_config = {"V100":8, "G6AiM":8, "Server": 2, "Rack_Unit": 8,"Require_Token_per_Second": Require_Token_per_Second,"Token_per_Second": 0}
    roofline_config = [{"Name":"1", "PRF":{"hard":"V100", "quant":8}, "GNR-PRJ":{"hard":"V100", "quant":8}, "GNR-ACT":{"hard":"G6AiM", "quant":8}, "Assign": "PRF+GNR-PRJ, GNR-ACT"}] 
    roofline = TransformerRoofline('hardware_models.json')
    roofline.Draw_Combination_Timebreakdown(Ratio_P=[0.25], Batchsize=[32], Max_Token=[8192], Model=["GPT3.5"], Comb_Hardware=roofline_config, Sparse=["default"])
    cost_config["Token_per_Second"] = roofline.Token_per_Sec[0]
    cost_model = CostEst(hardware_json="hardware_elements.json")
    cost_model.init_hardware(hardware_config=cost_config)
    cost_model.cal_total_cost()
    cost_model.List_Price_Breakdown(fig=True)

def test_1():
    """
    test_1: use cost model with pre-defined Token_per_Second for m*A100+n*G6AiM configuration
    """
    ## A100+G6AiM calculation for Xuhan
    Require_Token_per_Second = 99000000
    Hardware_Config = [
    {# Prefill:24*A100,GNR-PRF:24*A100,GNR-ACT:8*G6AiM
        "Config": "Prefill:24*A100,GNR-PRF:24*A100,GNR-ACT:8*G6AiM",
        "Roofline": [{"Name":"1", "PRF":{"hard":"A100", "quant":24}, "GNR-PRJ":{"hard":"A100", "quant":24}, "GNR-ACT":{"hard":"G6AiM", "quant":8}, "Assign": "PRF+GNR-PRJ, GNR-ACT"}],
        "Cost": {"A100":24, "G6AiM":8, "Server": 4, "Rack_Unit": 16,"Require_Token_per_Second": Require_Token_per_Second,"Token_per_Second": 454}
    },
    {# Prefill:16*A100,GNR-PRF:16*A100,GNR-ACT:16*G6AiM
        "Config": "Prefill:16*A100,GNR-PRF:16*A100,GNR-ACT:16*G6AiM",
        "Roofline": [{"Name":"1", "PRF":{"hard":"A100", "quant":16}, "GNR-PRJ":{"hard":"A100", "quant":16}, "GNR-ACT":{"hard":"G6AiM", "quant":16}, "Assign": "PRF+GNR-PRJ, GNR-ACT"}],
        "Cost": {"A100":16, "G6AiM":16, "Server": 4, "Rack_Unit": 16,"Require_Token_per_Second": Require_Token_per_Second,"Token_per_Second": 330}
    },
    {# Prefill:8*A100,GNR-PRF:8*A100,GNR-ACT:24*G6AiM
        "Config": "8*A100,GNR-PRF:8*A100,GNR-ACT:24*G6AiM",
        "Roofline": [{"Name":"1", "PRF":{"hard":"A100", "quant":8}, "GNR-PRJ":{"hard":"A100", "quant":8}, "GNR-ACT":{"hard":"G6AiM", "quant":24}, "Assign": "PRF+GNR-PRJ, GNR-ACT"}],
        "Cost": {"A100":8, "G6AiM":24, "Server": 4, "Rack_Unit": 16,"Require_Token_per_Second": Require_Token_per_Second,"Token_per_Second": 175}
    },
    {# Prefill:4*A100,GNR-PRF:28*G6AiM,GNR-ACT:28*G6AiM
        "Config": "4*A100,GNR-PRF:28*G6AiM,GNR-ACT:28*G6AiM",
        "Roofline": [{"Name":"1", "PRF":{"hard":"A100", "quant":4}, "GNR-PRJ":{"hard":"G6AiM", "quant":28}, "GNR-ACT":{"hard":"G6AiM", "quant":28}, "Assign": "PRF, GNR-PRJ+GNR-ACT"}],
        "Cost": {"A100":4, "G6AiM":28, "Server": 4, "Rack_Unit": 16,"Require_Token_per_Second": Require_Token_per_Second,"Token_per_Second": 593}
    },
    {# Prefill:8*A100,GNR-PRF:24*G6AiM,GNR-ACT:24*G6AiM
        "Config": "8*A100,GNR-PRF:24*G6AiM,GNR-ACT:24*G6AiM",
        "Roofline": [{"Name":"1", "PRF":{"hard":"A100", "quant":8}, "GNR-PRJ":{"hard":"G6AiM", "quant":24}, "GNR-ACT":{"hard":"G6AiM", "quant":24}, "Assign": "PRF, GNR-PRJ+GNR-ACT"}],
        "Cost": {"A100":8, "G6AiM":24, "Server": 4, "Rack_Unit": 16,"Require_Token_per_Second": Require_Token_per_Second,"Token_per_Second": 721}
    },
    {# Prefill:12*A100,GNR-PRF:20*G6AiM,GNR-ACT:20*G6AiM
        "Config": "12*A100,GNR-PRF:20*G6AiM,GNR-ACT:20*G6AiM",
        "Roofline": [{"Name":"1", "PRF":{"hard":"A100", "quant":12}, "GNR-PRJ":{"hard":"G6AiM", "quant":20}, "GNR-ACT":{"hard":"G6AiM", "quant":20}, "Assign": "PRF, GNR-PRJ+GNR-ACT"}],
        "Cost": {"A100":12, "G6AiM":20, "Server": 4, "Rack_Unit": 16,"Require_Token_per_Second": Require_Token_per_Second,"Token_per_Second": 699}
    },
    {# Prefill:16*A100,GNR-PRF:16*G6AiM,GNR-ACT:16*G6AiM
        "Config": "16*A100,GNR-PRF:16*G6AiM,GNR-ACT:16*G6AiM",
        "Roofline": [{"Name":"1", "PRF":{"hard":"A100", "quant":16}, "GNR-PRJ":{"hard":"G6AiM", "quant":16}, "GNR-ACT":{"hard":"G6AiM", "quant":16}, "Assign": "PRF, GNR-PRJ+GNR-ACT"}],
        "Cost": {"A100":16, "G6AiM":16, "Server": 4, "Rack_Unit": 16,"Require_Token_per_Second": Require_Token_per_Second,"Token_per_Second": 608}
    },
    {# Prefill:20*A100,GNR-PRF:12*G6AiM,GNR-ACT:12*G6AiM
        "Config": "20*A100,GNR-PRF:12*G6AiM,GNR-ACT:12*G6AiM",
        "Roofline": [{"Name":"1", "PRF":{"hard":"A100", "quant":20}, "GNR-PRJ":{"hard":"G6AiM", "quant":12}, "GNR-ACT":{"hard":"G6AiM", "quant":12}, "Assign": "PRF, GNR-PRJ+GNR-ACT"}],
        "Cost": {"A100":20, "G6AiM":12, "Server": 4, "Rack_Unit": 16,"Require_Token_per_Second": Require_Token_per_Second,"Token_per_Second": 482}
    }
    ]

    cost_model = CostEst(hardware_json="hardware_elements.json")

    config = []
    price_per_token = []
    for hw in Hardware_Config:
        cost_model.init_hardware(hardware_config=hw["Cost"])
        cost_model.cal_total_cost()
        print("------------ Hardware config: {} -------------".format(hw["Config"]))
        print("Token per Sec: {}".format(hw["Cost"]["Token_per_Second"]))
        print("Price per Token: {}".format(cost_model.price_per_token))
        print("----------------------------------------------")
        config.append(hw["Config"])
        price_per_token.append(cost_model.price_per_token)

    plt.figure(figsize=(6,6))
    plt.barh(config, price_per_token)
    for index, y_value in enumerate(price_per_token):
        plt.text(y_value*1.05, index-0.2, "%.2e"%price_per_token[index])
    plt.xlabel("Price per Token ($)")
    plt.ylabel("Configuration")
    plt.show() 

def test_2():
    """
    test_2: 
    use cost model with roofline model;
    Token_per_Second calculated by Xuhan's roofline model
    """
    ## Select 4 typical hardware for comparison
    Require_Token_per_Second = 99000000
    Hardware_Config = [
        {# 8*A100
            "Config": "8*A100",
            "Roofline": [{"Name":"1", "PRF":{"hard":"A100", "quant":8}, "GNR-PRJ":{"hard":"A100", "quant":8}, "GNR-ACT":{"hard":"A100", "quant":8}, "Assign": "PRF+GNR-PRJ+GNR-ACT"}],
            "Cost": {"A100":8, "Server": 1, "Rack_Unit": 4,"Require_Token_per_Second": Require_Token_per_Second,"Token_per_Second": 0}
        },
        {# 8*A100 + 8*G6AiM
            "Config": "8*A100 + 8*G6AiM",
            "Roofline": [{"Name":"1", "PRF":{"hard":"A100", "quant":8}, "GNR-PRJ":{"hard":"A100", "quant":8}, "GNR-ACT":{"hard":"G6AiM", "quant":8}, "Assign": "PRF+GNR-PRJ, GNR-ACT"}],
            "Cost": {"A100":8, "G6AiM":8, "Server": 2, "Rack_Unit": 8,"Require_Token_per_Second": Require_Token_per_Second,"Token_per_Second": 0}
        },
        {# 8*A100 + 8*LPDDR4-PIM-DIMM
            "Config": "8*A100 + 8*LPDDR4-PIM-DIMM",
            "Roofline": [{"Name":"1", "PRF":{"hard":"A100", "quant":8}, "GNR-PRJ":{"hard":"A100", "quant":8}, "GNR-ACT":{"hard":"LPDDR4-PIM-DIMM", "quant":8}, "Assign": "PRF+GNR-PRJ, GNR-ACT"}],
            "Cost": {"A100":8, "LPDDR4-PIM-DIMM":8, "Server": 2, "Rack_Unit": 8,"Require_Token_per_Second": Require_Token_per_Second,"Token_per_Second": 0}
        },
        {# 8*THRIVE2 + 8*LPDDR4-PIM-DIMM
            "Config": "8*THRIVE2 + 8*LPDDR4-PIM-DIMM",
            "Roofline": [{"Name":"1", "PRF":{"hard":"THRIVE2", "quant":8}, "GNR-PRJ":{"hard":"THRIVE2", "quant":8}, "GNR-ACT":{"hard":"LPDDR4-PIM-DIMM", "quant":8}, "Assign": "PRF+GNR-PRJ, GNR-ACT"}],
            "Cost": {"THRIVE2":8, "LPDDR4-PIM-DIMM":8, "Server": 2, "Rack_Unit": 8,"Require_Token_per_Second": Require_Token_per_Second,"Token_per_Second": 0}
        }
    ]

    roofline = TransformerRoofline('hardware_models.json')
    cost_model = CostEst(hardware_json="hardware_elements.json")

    for hw in Hardware_Config:
        roofline.Draw_Combination_Timebreakdown(Ratio_P=[0.25], Batchsize=[32], Max_Token=[8192], Model=["GPT3.5"], Comb_Hardware=hw["Roofline"], Sparse=["default"])
        hw["Cost"]["Token_per_Second"] = roofline.Token_per_Sec[0]

    i = 1
    plt.figure(figsize=(15,15))
    for hw in Hardware_Config:
        cost_model.init_hardware(hardware_config=hw["Cost"])
        cost_model.cal_total_cost()
        if hw["Config"] == "8*A100":
            price_per_token_base = cost_model.price_per_token
        price = [
            cost_model.XPU_price, 
            cost_model.PIM_price, 
            cost_model.operation_price_XPU, 
            cost_model.operation_price_PIM, 
            cost_model.operation_price_server, 
            cost_model.server_total_price, 
            cost_model.datacenter_op_fee
        ]
        labels = ["XPU", "PIM", "op_XPU", "op_PIM", "op_Server", "Server", "Others"]
        plt.subplot(2,2,i)
        plt.pie(price, labels=labels, autopct="%1.1f%%", textprops={'fontsize':25})
        plt.title(hw["Config"], fontdict={'fontsize':25})
        i+=1
    plt.show()

if __name__ == "__main__":
    # test_0_0()
    test_0_1()
    # test_0_2()
    # test_1()
    # test_2()
'''