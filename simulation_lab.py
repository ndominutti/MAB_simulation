import numpy as np
import pandas as pd
from mab_methods import BernoulliThompsonSampling, BernoulliEpsilonGreedy, BernoulliUCB
from collections import Counter
from scipy.stats import beta
import copy
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


class SimulationLab:
    def __init__(
        self,
        actual_probs: list,
        repetitions: int,
        iterations: int,
        epsilon: float,
        random_seed: int = 13,
    ):
        """Simulation environment for the usage of MAB sampling methods.

        Args:
            actual_probabilities (list): real underlying probabilities of success for
            each slot machine
            repetitions (int): amount of repetitions of iteration trials
            iterations (int): amount of steps in a single repetition
            epsilon (float): epsilon argument for epsilon-greedy algorithm
            random_seed (int, optional): seed for reproducibility. Defaults to 13.
        """
        self.K = len(actual_probs)
        self.actual_probs = actual_probs
        self.repetitions = repetitions
        self.iterations = iterations
        self.epsilon = epsilon
        np.random.seed(random_seed)

    def _initialize(self):
        """Start the process by creating empty arrays that will be used in next step.
        The shape of the arrays is repetitions*iterations, so every repetition will be stored
        in a row, so in further steps we will take a mean column wise.
        """
        self.TSGlobalRegret = np.zeros((self.repetitions, self.iterations))
        self.TSGlobalHistorical = np.zeros((self.repetitions, self.iterations))
        self.TSGlobalMABselection = np.zeros((self.repetitions, self.iterations))
        self.eGFGlobalRegret = np.zeros((self.repetitions, self.iterations))
        self.eGFGlobalMABselection = np.zeros((self.repetitions, self.iterations))
        self.eGFGlobalHistorical = np.zeros((self.repetitions, self.iterations))
        self.eGTGlobalRegret = np.zeros((self.repetitions, self.iterations))
        self.eGTGlobalMABselection = np.zeros((self.repetitions, self.iterations))
        self.eGTGlobalHistorical = np.zeros((self.repetitions, self.iterations))
        self.UCBGlobalRegret = np.zeros((self.repetitions, self.iterations))
        self.UCBGlobalHistorical = np.zeros((self.repetitions, self.iterations))
        self.UCBGlobalMABselection = np.zeros((self.repetitions, self.iterations))

    def _simulation(self):
        """Simulation environment. For each repetition, it stores the regret, historical and
        machine selection for every model in every iteration and storing them in the empty arrays
        created with _initialize method.
        """
        for rep in tqdm(range(self.repetitions)):
            TS = BernoulliThompsonSampling(self.K)
            TSRegret = []
            TSHistorical = []
            TSMABselection = []

            eGF = BernoulliEpsilonGreedy(self.K, epsilon=self.epsilon, decay=False)
            eGFRegret = []
            eGFMABselection = []
            eGFHistorical = []

            eGT = BernoulliEpsilonGreedy(self.K, epsilon=self.epsilon, decay=True)
            eGTRegret = []
            eGTMABselection = []
            eGTHistorical = []

            UCB = BernoulliUCB(self.K)
            UCBRegret = []
            UCBHistorical = []
            UCBMABselection = [*range(self.K)]

            for i in range(self.iterations):
                TSmab_selection = TS.select_mab()
                TSMABselection.append(TSmab_selection)
                eGTmab_selection = eGT.select_mab(i + 1)
                eGTMABselection.append(eGTmab_selection)
                eGFmab_selection = eGF.select_mab()
                eGFMABselection.append(eGFmab_selection)
                UCBmab_selection = UCB.select_mab(i + 1, UCBMABselection)
                UCBMABselection.append(UCBmab_selection)

                # Single trial for every slot machine
                real_outcome = [
                    TS.sample_from_real_distribution(self.actual_probs, MAB)
                    for MAB in range(self.K)
                ]

                TSreward = real_outcome[TSmab_selection]
                eGTreward = real_outcome[eGTmab_selection]
                eGFreward = real_outcome[eGFmab_selection]
                UCBreward = real_outcome[UCBmab_selection]

                TS.update_posterior(TSmab_selection, TSreward)
                TSHistorical.append(TSreward)
                TSRegret.append(
                    TS.calculate_step_regret(self.actual_probs, i, TSHistorical)
                )
                eGT.update_posterior(eGTmab_selection, i, eGTreward)
                eGTHistorical.append(eGTreward)
                eGTRegret.append(
                    eGT.calculate_step_regret(self.actual_probs, i, eGTHistorical)
                )
                eGF.update_posterior(eGFmab_selection, i, eGFreward)
                eGFHistorical.append(eGFreward)
                eGFRegret.append(
                    eGF.calculate_step_regret(self.actual_probs, i, eGFHistorical)
                )
                UCB.update_posterior(UCBmab_selection, i, UCBreward)
                UCBHistorical.append(UCBreward)
                UCBRegret.append(
                    UCB.calculate_step_regret(self.actual_probs, i, UCBHistorical)
                )

            print(f"Saving repetition {rep}...")
            self.TSGlobalRegret[rep, :] = np.array(TSRegret)
            self.TSGlobalHistorical[rep, :] = np.array(TSHistorical)
            self.TSGlobalMABselection[rep, :] = np.array(TSMABselection)

            self.eGFGlobalRegret[rep, :] = np.array(eGFRegret)
            self.eGFGlobalHistorical[rep, :] = np.array(eGFHistorical)
            self.eGFGlobalMABselection[rep, :] = np.array(eGFMABselection)

            self.eGTGlobalRegret[rep, :] = np.array(eGTRegret)
            self.eGTGlobalHistorical[rep, :] = np.array(eGTHistorical)
            self.eGTGlobalMABselection[rep, :] = np.array(eGTMABselection)

            self.UCBGlobalRegret[rep, :] = np.array(UCBRegret)
            self.UCBGlobalHistorical[rep, :] = np.array(UCBHistorical)
            self.UCBGlobalMABselection[rep, :] = np.array(
                UCBMABselection[len(self.actual_probs) :]
            )

    def _machine_selection_evolution(self):
        """Uses the global MAB selection array to get an historical of how each method
        selected the machines.
        """
        empty_df_template = pd.DataFrame(
            {
                "iteration": [0] * self.K,
                "accumulated_selection": [0] * self.K,
                "slot_machine_n": [*range(self.K)],
            }
        )
        empty_template = {
            "iteration": [],
            "accumulated_selection": [],
            "slot_machine_n": [],
        }
        iterable_dict = {
            "TS": [
                copy.deepcopy(empty_df_template),
                self.TSGlobalMABselection,
                copy.deepcopy(empty_template),
            ],
            "egT": [
                copy.deepcopy(empty_df_template),
                self.eGTGlobalMABselection,
                copy.deepcopy(empty_template),
            ],
            "egF": [
                copy.deepcopy(empty_df_template),
                self.eGFGlobalMABselection,
                copy.deepcopy(empty_template),
            ],
            "UCB": [
                copy.deepcopy(empty_df_template),
                self.UCBGlobalMABselection,
                copy.deepcopy(empty_template),
            ],
        }

        for model in tqdm(iterable_dict.keys()):
            for i in range(self.iterations):
                iterable_dict[model][2] = copy.deepcopy(empty_template)
                for key in range(self.K):
                    iterable_dict[model][2]["slot_machine_n"].append(key)
                    iterable_dict[model][2]["accumulated_selection"].append(
                        iterable_dict[model][0][
                            iterable_dict[model][0]["slot_machine_n"] == key
                        ].accumulated_selection.values[-1]
                    )
                    for rep in range(self.repetitions):
                        if iterable_dict[model][1][rep, i] == key:
                            iterable_dict[model][2]["accumulated_selection"][-1] = (
                                iterable_dict[model][2]["accumulated_selection"][-1] + 1
                            )
                    iterable_dict[model][2]["iteration"].append(i)
                iterable_dict[model][0] = pd.concat(
                    [iterable_dict[model][0], pd.DataFrame(iterable_dict[model][2])]
                )
                iterable_dict[model][0]["model"] = model

        self.count_df = pd.concat([model[0] for model in iterable_dict.values()])
        self.count_df["slot_machine_n"] = (
            self.count_df.slot_machine_n.astype("int").astype("str")
            + "_"
            + self.count_df.model
        )
        self.count_df["accumulated_selection"] = (
            self.count_df["accumulated_selection"] / self.repetitions
        )

    def simulate(self):
        """ """
        print("Starting simulation...")
        self._initialize()
        self._simulation()
        print("Success!")
        print("Starting machine selection evolution calculus...")
        self._machine_selection_evolution()
        print("Success!")

    def plot_mab_selection(self):
        """ """
        fig = px.bar(
            self.count_df[self.count_df.iteration % 10 == 0],
            x="slot_machine_n",
            y="accumulated_selection",
            color="model",
            animation_frame="iteration",
            title="Evolución de las máquinas elegidas en función de t (AVG de todas las repeticiones)",
            labels={
                "slot_machine_n": "máquinas",
                "accumulated_selection": "promedio de seleccion acumulada",
            },
        )
        fig.update_layout(yaxis_range=[0, self.iterations])
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 0.01
        return fig

    def plot_regret(self):
        """_summary_"""
        self.UCBRegret = self.UCBGlobalRegret.mean(axis=0)
        self.eGTRegret = self.eGTGlobalRegret.mean(axis=0)
        self.eGFRegret = self.eGFGlobalRegret.mean(axis=0)
        self.TSRegret = self.TSGlobalRegret.mean(axis=0)

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 12))
        sns.lineplot(
            x=range(1, self.iterations + 1),
            y=np.array(self.TSRegret) / range(1, self.iterations + 1),
            label="TS",
            color="darkblue",
            alpha=0.7,
            ax=axes[0],
        )
        sns.lineplot(
            x=range(1, self.iterations + 1),
            y=np.array(self.UCBRegret) / range(1, self.iterations + 1),
            label="UCB",
            color="purple",
            alpha=0.7,
            ax=axes[0],
        )
        sns.lineplot(
            x=range(1, self.iterations + 1),
            y=np.array(self.eGTRegret) / range(1, self.iterations + 1),
            label="epsilon-greedy (con epsilon-decay)",
            color="red",
            alpha=0.7,
            ax=axes[0],
        )
        sns.lineplot(
            x=range(1, self.iterations + 1),
            y=np.array(self.eGFRegret) / range(1, self.iterations + 1),
            label="epsilon-greedy (sin epsilon-decay)",
            color="green",
            alpha=0.7,
            ax=axes[0],
        )
        axes[0].legend()
        axes[0].axhline(y=0, xmin=0, xmax=2000, color="black", ls="--", alpha=0.3)
        axes[0].set_title("Convergencia de Regret(n)/n")

        sns.lineplot(
            x=range(1, self.iterations + 1),
            y=np.array(self.TSRegret) / range(1, self.iterations + 1),
            label="TS",
            color="darkblue",
            alpha=0.7,
            ax=axes[1],
        )
        sns.lineplot(
            x=range(1, self.iterations + 1),
            y=np.array(self.UCBRegret) / range(1, self.iterations + 1),
            label="UCB",
            color="purple",
            alpha=0.7,
            ax=axes[1],
        )
        sns.lineplot(
            x=range(1, self.iterations + 1),
            y=np.array(self.eGTRegret) / range(1, self.iterations + 1),
            label="epsilon-greedy (con epsilon-decay)",
            color="red",
            alpha=0.7,
            ax=axes[1],
        )
        sns.lineplot(
            x=range(1, self.iterations + 1),
            y=np.array(self.eGFRegret) / range(1, self.iterations + 1),
            label="epsilon-greedy (sin epsilon-decay)",
            color="green",
            alpha=0.7,
            ax=axes[1],
        )
        axes[1].legend()
        axes[1].axhline(y=0, xmin=0, xmax=2000, color="black", ls="--", alpha=0.3)
        axes[1].set_title("Convergencia de Regret(n)/n (zoom in)")
        axes[1].set_xlim((self.iterations * 0.5, self.iterations))
        axes[1].set_ylim((-0.02, 0.1))

    def plot_log_comparison(self):
        """_summary_"""
        plt.figure(figsize=(18, 6))
        sns.lineplot(x=np.sqrt(range(self.iterations)), y=self.TSRegret, label="TS")
        sns.lineplot(
            x=np.sqrt(range(self.iterations)),
            y=self.eGTRegret,
            label="EG (con epsilon-decay)",
            color="red",
        )
        sns.lineplot(
            x=np.sqrt(range(self.iterations)),
            y=self.eGFRegret,
            label="EG (sin epsilon-decay)",
            color="green",
        )
        sns.lineplot(
            x=np.sqrt(range(self.iterations)),
            y=self.UCBRegret,
            label="UCB",
            color="purple",
        )
        plt.plot(
            np.sqrt(range(self.iterations)),
            np.sqrt(range(self.iterations)),
            label="45°",
            color="black",
            ls="--",
        )
        plt.ylabel("Rn")
        plt.xlabel("sqrt(n)")
        plt.title("Comparación velocidad de crecimiento regret vs sqrt(n)")
        plt.legend()
