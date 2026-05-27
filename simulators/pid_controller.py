import numpy as np


class PIDController:
    """
    Stateless (bandit) simulation of a PID controller that controls the the pump
    speed during the backwashing phase in the water treatment plant
    """
    def __init__(self,
                 setpoint=0.6309013,
                 backwashing_total_time=55,
                 initial_pump_speed=20,
                 ideal_flow_rate=None) -> None:
        self.setpoint = setpoint
        self.backwashing_total_time = backwashing_total_time
        self.initial_pump_speed = initial_pump_speed
        if ideal_flow_rate:
            self.ideal_flow_rate = ideal_flow_rate
        else:
            self.ideal_flow_rate = [0,  0.15,  0.3, 0.4, 0.46, 0.49, 0.52, 0.55, 0.58, 0.60, 0.61, 0.62, setpoint]
            for t in range(self.backwashing_total_time - len(self.ideal_flow_rate)):
                self.ideal_flow_rate.append(setpoint)
        self.ideal_flow_rate = np.asarray(self.ideal_flow_rate)

    def __get_inst_simulated_flow_rate(self, pump_speed: float) -> float:
        """
        returns instantaneous flow_rate for a given pump speed by modelling the
        relationship between pump_speed and flow_rate using real data
        """
        p = np.array([ 0.00060553, -0.02530755,  0.26727398])
        return p[0]*pump_speed**2 + p[1]*pump_speed + p[2] + np.random.normal(0, 0.003)

    def __get_pid_controlled_flow_rate(self, k_p: float, k_i: float, k_d: float) -> np.ndarray:
        """
        adjust the pump speed using a PID controller, return the simulated
        flow rate over time
        """
        simulated_flow_rates = []
        errors = [0]
        pump_speed = self.initial_pump_speed
        dt=1
        # note: following assumes constant time intervals
        for t in range(0, self.backwashing_total_time):
            flow_rate = self.__get_inst_simulated_flow_rate(pump_speed)
            errors.append(self.setpoint - flow_rate)
            pump_speed += k_p*errors[-1] + k_i*np.sum(errors)*dt + k_d*(errors[-1] - errors[-2])/dt
            simulated_flow_rates.append(flow_rate)
        return np.asarray(simulated_flow_rates)

    def __get_reward(self, ideal_flow_rate: np.ndarray, simulated_flow_rate: np.ndarray) -> float:
        """
        returns euclidean distance between the ideal and simulated flow rates
        """
        reward = - np.linalg.norm(ideal_flow_rate - simulated_flow_rate)
        return float(reward)

    def step(self, action: list[float]) -> float:
        """
        action: PID controller parameters [k_p, k_i, k_d]
        """
        flow_rate = self.__get_pid_controlled_flow_rate(*action)
        return self.__get_reward(self.ideal_flow_rate, flow_rate)
