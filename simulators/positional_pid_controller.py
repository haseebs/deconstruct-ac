import numpy as np

# TODO Change the ideal flow rate to trying to match the setpoint as quickly as possible
class PositionalPIDController:
    """
    Stateless (bandit) simulation of a PID controller that controls the the pump
    speed during the backwashing phase in the water treatment plant
    """
    def __init__(self,
                 setpoint=0.6309013,
                 backwashing_total_time=55,
                 initial_pump_speed=20,
                 reward_noise=0.00) -> None:
        self.setpoint = setpoint
        self.backwashing_total_time = backwashing_total_time
        self.initial_pump_speed = initial_pump_speed
        self.ideal_flow_rate = np.asarray([setpoint]*backwashing_total_time)

        self.reward_noise = reward_noise
        self.MAX_PUMP_SPEED = 100
        self.MIN_PUMP_SPEED = 0


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
            pump_speed = k_p*errors[-1] + k_i*np.sum(errors)*dt + k_d*(errors[-1] - errors[-2])/dt # correct positional
            # clip the pump speed between [MAX and MIN]
            if pump_speed > self.MAX_PUMP_SPEED:
                pump_speed = self.MAX_PUMP_SPEED
            elif pump_speed < self.MIN_PUMP_SPEED:
                pump_speed = self.MIN_PUMP_SPEED
            simulated_flow_rates.append(flow_rate)
        return np.asarray(simulated_flow_rates)

    def __get_reward(self, ideal_fl15ow_rate: np.ndarray, simulated_flow_rate: np.ndarray) -> float:
        """
        returns euclidean distance between the ideal and simulated flow rates
        """
        ideal_flow_rate = self.ideal_flow_rate
        vec = []
        for i in range(len(simulated_flow_rate)):
            if simulated_flow_rate[i] > 1.05 * ideal_flow_rate[i]:
                vec.append(5 * (simulated_flow_rate - ideal_flow_rate))
            else:
                vec.append(simulated_flow_rate - ideal_flow_rate)
        reward = - np.linalg.norm(vec)  
        reward = reward / 110
        reward += np.random.normal(0, self.reward_noise) # add noise
        return float(reward)

    def step(self, action: list[float]) -> float:
        """
        action: PID controller parameters [k_p, k_i, k_d]
        """
        flow_rate = self.__get_pid_controlled_flow_rate(*action)
        return self.__get_reward(self.ideal_flow_rate, flow_rate)
