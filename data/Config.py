import importlib
import numpy as np
import torch
from algorithm.neuron import *


class Config:
    def __init__(
        self, data_set_name, time_step_eq=None, settings_file=None
    ):
        """
        settings_file: name of file in algorithm/config/settings
        """

        assert settings_file is not None, "please provide settings"


        settings = self.get_settings(settings_file)

        self.load_parameters(data_set_name, settings)

        print(f'self.synapse intr noise is {self.synapse_noise}')

        self.init_tau_mem=self.tau_mem

        # if time_step_eq is not None:
        # self.get_eq_taus_(time_step_eq)
        self.set_general_parameters()


    def get_settings(self, settings_file):
        settings = importlib.import_module(
            f"algorithm.config.settings.{settings_file}"
        )
        return settings

    def load_parameters(self, data_set_name, settings):
        data_set_names = {"shd", "mnist", "rm","rms","rml"}
        data_set_name = (
            "other" if data_set_name not in data_set_names else data_set_name
        )
        exclude_list = {
            "torch",
            "LowerBoundL2",
            "UpperBoundL2",
            "UpperBoundL1",
            "almost_xavier_uniform_",
            "almost_xavier_normal_",
        }
        self.__dict__.update(
            (k, v)
            for (k, v) in settings.__dict__.items()
            if ((not k.startswith("__")) and (not k in exclude_list))
        )
        if "specifics" in self.__dict__.keys():

            self.__dict__.update(
                (k, v)
                for (k, v) in settings.__dict__["specifics"][data_set_name].items()
            )
    

    def set_general_parameters(self):
        # general LIF parameters
        
        self.dtype = torch.float
        self.device =torch.device("cpu") #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net_size = [
            self.batch_size,
            self.nb_inputs,
            self.nb_hidden,
            self.nb_outputs,
        ]
        print(f'nb hidden inside config: {self.nb_hidden}')

        
        self.set_neuron_synapse_params()
        self.set_neuron_synapse_clipping()
        self.set_weight_params()
        self.set_regularizers()

        print(f'shape in config tau_mem: {self.tau_mem.shape}')
        print(f'shape in config beta: {self.beta.shape}')
        print(f'config wsf: {self.weight_scaling_factor}')

    def set_weight_params(self):
        self.weight_info = self.train_weights,self.Dales_law,self.weight_init_,self.target_weights,self.weight_scaling_factor,self.recurrent_sparse,self.low_rank_sparse

    def set_neuron_synapse_params(self):
        
        #set distribution over parameters
        if self.set_I_c_range:
            self.I_c=np.linspace(-self.I_c_abs, self.I_c_abs, self.nb_hidden)
            np.random.shuffle(self.I_c)
        else:
            self.I_c=np.repeat(self.I_c_abs, self.nb_hidden)
        self.I_c=torch.tensor(self.I_c)

        if self.set_tau_mem_range:
            self.tau_mem = np.linspace(self.init_tau_mem/2, self.init_tau_mem*2, self.nb_hidden)
            np.random.shuffle(self.tau_mem)
        else:
            #pass #already set
            if not isinstance(self.tau_mem, (list, np.ndarray,torch.Tensor)) or (len(self.tau_mem)!=self.nb_hidden):
                self.tau_mem = np.repeat(self.init_tau_mem, self.nb_hidden) #always use as array?
        self.tau_mem=torch.tensor(self.tau_mem)

        # only depends on provided parameters
        self.alpha = torch.exp(-self.time_step / torch.tensor(self.tau_syn))
        self.beta = torch.exp(-self.time_step / torch.tensor(self.tau_mem))
        self.alpha_out = torch.exp(-self.time_step / torch.tensor(self.tau_syn_out))
        self.beta_out = torch.exp(-self.time_step / torch.tensor(self.tau_mem_out))

        if self.neuron_name=='LIF':
            self.neuron_fct=LIF
            self.neuron_intrinsic=[self.V_rest,self.tau_mem,self.tau_mem_out,self.V_thresh,self.V_reset,None,self.I_c]
        elif self.neuron_name=='BLK_nonsp':
            self.neuron_fct=BLK_nonsp
            self.neuron_intrinsic=[self.V_rest,self.tau_mem,self.tau_mem_out,None,self.V_reset,None,self.I_c]
        elif 'QIF' in self.neuron_name: #any of the QIF models
                self.neuron_intrinsic=[self.V_sn,self.tau_mem,self.tau_mem_out,self.V_thresh,self.V_reset,self.a,self.I_c]

        print(f'self.synapse_noise: {self.synapse_noise}')
        self.synapse_intrinsic = [self.synapse, self.tau_syn, self.tau_syn_out, self.synapse_noise]

    def match_QIF_to_LIF(self):
            div_SNIC = -self.div_factor
            div_HOM = self.div_factor
            if 'SNIC' in self.neuron_name:
                QIF_o = match_QIF_params(
                    self.tau_mem,
                    V_sn=self.V_sn,
                    div=div_SNIC, 
                    V_rest_LIF=self.V_rest,
                    V_reset_LIF=self.V_reset,
                    V_thresh_LIF=self.V_thresh,
                )
            elif 'HOM' in self.neuron_name:
                QIF_o = match_QIF_params(
                    self.tau_mem,
                    V_sn=self.V_sn,
                    div=div_HOM,  # +!!
                    V_rest_LIF=self.V_rest,
                    V_reset_LIF=self.V_reset,
                    V_thresh_LIF=self.V_thresh,
                )

            elif 'other' in self.neuron_name:
                if self.other_reset is None:
                    self.other_reset = 0.5
                QIF_o = match_QIF_params(
                    settings.tau_mem,
                    V_sn=self.V_sn,
                    V_reset_QIF=self.other_reset,  
                    V_rest_LIF=self.V_rest,
                    V_reset_LIF=self.V_reset,
                    V_thresh_LIF=self.V_thresh,
                )
            elif 'mixed' in self.neuron_name:
                
                QIF_o = self.get_mixed_QIF(perc=perc_SNIC)

            return QIF_o

    def get_mixed_QIF(self,perc=0.5):
        # perc: proportion of SNIC neurons
        div_SNIC = -self.div_factor
        div_HOM = self.div_factor
        QIF_SNIC = match_QIF_params(
            self.tau_mem,
            V_sn=self.V_sn,
            div=div_SNIC,  # -!!
            V_rest_LIF=self.V_rest,
            V_reset_LIF=self.V_reset,
            V_thresh_LIF=self.V_thresh,
        )
        QIF_HOM = match_QIF_params(
            self.tau_mem,
            V_sn=V_sn,
            div=div_HOM,  # +!!
            V_rest_LIF=self.V_rest,
            V_reset_LIF=self.V_reset,
            V_thresh_LIF=self.V_thresh,
        )
        nb_cut = int(perc * self.nb_hidden)

        V_resets_mixed = torch.zeros((self.nb_hidden))
        V_resets_mixed[:nb_cut] = QIF_SNIC.V_reset
        V_resets_mixed[nb_cut:] = QIF_HOM.V_reset
        V_sn_mixed = torch.full((self.nb_hidden, 1), QIF_SNIC.V_sn)
        tau_mems_mixed = torch.zeros((self.nb_hidden))
        tau_mems_mixed[:nb_cut] = QIF_SNIC.tau_mem[:nb_cut]
        tau_mems_mixed[nb_cut:] = QIF_HOM.tau_mem[nb_cut:]
        I_c_mixed = torch.full((self.nb_hidden, 1), QIF_SNIC.I_c)
        th_mixed = torch.zeros((self.nb_hidden))
        th_mixed[:nb_cut] = QIF_SNIC.V_peak[:nb_cut]
        th_mixed[nb_cut:] = QIF_HOM.V_peak[nb_cut:]
        a_mixed = torch.full((self.nb_hidden, 1), QIF_SNIC.a)

        QIF_o = QIF_cont(
            V_sn=V_sn_mixed,
            V_reset=V_resets_mixed,
            tau_mem=tau_mems_mixed,
            V_peak=th_mixed,
            a=a_mixed,
            I_c=I_c_mixed,
        )
        return QIF_o

    def set_neuron_synapse_clipping(self):

        # clipping ranges for parameters do not depend on other settings
        reset_range = [-1.0, 1.0]
        th_range = [0.0, 3.0]
        inverse_tau_range = [np.exp(-1 / 3),
            0.995,
        ] # from Perez Nieves et al. (2021)

        I_c_range = [-self.I_c_abs, self.I_c_abs]
        self.clip_intrinsic = {  # only used if parameter mentioned in train_intrinsic
            "reset": reset_range,
            "alpha": inverse_tau_range,
            "beta": inverse_tau_range,
            "th": th_range,
            "I_c": I_c_range
        }

    def set_regularizers(self):
        if self.reg:
            # lower bound: per neuron
            self.regLB = self.LowerBound(
                self.lower_strength, threshold=self.lower_thresh, dims=False
            )
            # upper bound: population
            self.regUB = self.UpperBound(
                self.upper_strength,
                threshold=self.upper_thresh,
                dims=1,  # -> after summing over time: dim 1 is nr of neurons
            )

            self.regularizers = [self.regLB, self.regUB]
        else:
            self.regularizers = None



'''
    def get_eq_taus_(self, time_step_eq):
        """
        to create an LIF that is equivalent to the one with the original time step,
        adapt membrane and synapse time constants (and time step) accordingly
        new_tau = old_tau * (new_dt/old_dt)

        arguments:
            time_step_eq: float
                new time step that equivalent time constants should be set for
        """
        if self.time_step != time_step_eq:

            ratio_dts = time_step_eq / self.time_step

            self.tau_mem *= ratio_dts
            self.tau_syn *= ratio_dts

            self.time_step = time_step_eq
            
'''