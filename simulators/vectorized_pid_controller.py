import numpy as np


class VectorizedPIDController:
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
        p = np.array([ 0.00060553, -0.02530755,  0.26727398])
        return p[0]*pump_speed**2 + p[1]*pump_speed + p[2] + np.random.normal(0, 0.003)

    def __get_pid_controlled_flow_rate(self, action: np.ndarray) -> np.ndarray:
        """
        make PID controller vectorized, take in a vector of actions and 
        output a matrix whose rows are corresponding flow sequences
        """
        simulated_flow_rates = np.zeros((action.shape[0], self.backwashing_total_time))
        errors = [np.zeros((action.shape[0], ))]
        pump_speed = self.initial_pump_speed * np.ones((action.shape[0], ))
        dt=1
        # note: following assumes constant time intervals
        for t in range(0, self.backwashing_total_time):
            flow_rate = self.__get_inst_simulated_flow_rate(pump_speed)
            errors.append((self.setpoint - flow_rate).squeeze())
            pump_speed += action[:, 0]*errors[-1] + action[:, 1]*np.sum(errors, axis=0)*dt + action[:, 2]*(errors[-1] - errors[-2])/dt
            simulated_flow_rates[:, t] = flow_rate
        return simulated_flow_rates

    def __get_reward(self, ideal_flow_rate: np.ndarray, simulated_flow_rate: np.ndarray) -> float:
        """note that the simulated flow rate is a matrix whose rows are flow rate sequences"""
        reward = - np.sqrt(np.sum(np.abs(ideal_flow_rate - simulated_flow_rate)**2, 1))
        return reward

    def step(self, action: np.ndarray) -> np.ndarray:
        flow_rate = self.__get_pid_controlled_flow_rate(action)
        return self.__get_reward(self.ideal_flow_rate, flow_rate)
