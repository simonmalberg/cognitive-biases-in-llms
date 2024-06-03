import numpy as np


class AnchoringBiasMetric:
    """
    A class that describes the quantitative evaluation of the anchoring bias in a model.
    Currently, two variations of individual metric are implemented: anchor agnostic and anchor specific,

    Individual agnostic:
    𝔅 = ‖ â₂ − â₁ ‖₁ / ‖ â₁ − ã₁ ‖₁ ∈ [0, 1];

    Individual specific:
    𝔅' = max[0, (‖ â₁ − a' ‖₁ − ‖ â₂ − a' ‖₁) / ‖ â₁ − a' ‖₁] ∈ [0, 1];

    Batch metric:
    𝔅 = (𝔅₁ + ... + 𝔅ₙ) / n,

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    ã₁ is the farthest answer option from â₁;
    a' is the answer option closest to the anchor value;
    n is number of test cases in the batch.

    Attributes:
        anchor_agnostic (bool): A flag that is used for choice between two variations of the metric.
        overall (bool): A flag that is used to indicate that a single result per batch of test is required.

    """

    def __init__(self, anchor_agnostic: bool, overall: bool):
        self.anchor_agnostic = anchor_agnostic
        self.overall = overall

    def anchor_agnostic_metric(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        answer_options: np.array,
    ) -> np.array:
        """
        Computes the anchor agnostic anchoring bias metric for the given batch of test instances.

        Args:
            control_answer (np.array, shape (batch, 1)): The answer(s) chosen for the control variant.
            treatment_answer (np.array, shape (batch, 1)): The answer(s) chosen for the treatment variant.
            answer_options (np.array, shape (batch, num_options)): The answer options available in the test case.

        Returns:
            The anchor agnostic anchoring bias metric value.
        """
        farthest_idx = np.argmax(
            np.abs(answer_options - control_answer), axis=1, keepdims=True
        )
        farthest_val = np.take_along_axis(answer_options, farthest_idx, axis=1)
        result = np.abs(control_answer - treatment_answer) / np.abs(
            control_answer - farthest_val + 1e-8
        )

        return result

    def anchor_specific_metric(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        answer_options: np.array,
        anchor: np.array,
    ) -> np.array:
        """
        Computes the anchor specific anchoring bias metric for the given batch of test instances.

        Args:
            control_answer (np.array, shape (batch, 1)): The answer(s) chosen for the control variant.
            treatment_answer (np.array, shape (batch, 1)): The answer(s) chosen for the treatment variant.
            answer_options (np.array, shape (batch, num_options)): The answer options available in the test case.
            anchor (np.array, shape (batch, 1)): The anchor value(s) used in the test case.

        Returns:
            The anchor specific anchoring bias metric value.
        """
        closest_idx = np.argmin(np.abs(anchor - answer_options), axis=1, keepdims=True)
        closest_val = np.take_along_axis(answer_options, closest_idx, axis=1)
        result = (
            np.abs(control_answer - closest_val)
            - np.abs(treatment_answer - closest_val)
        ) / np.abs(control_answer - closest_val + 1e-8)
        result = np.maximum(0, result)

        return result

    def compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        answer_options: np.array,
        anchor: np.array,
    ) -> np.array:
        """
        Computes the chosen anchoring bias metric.

        Args:
            control_answer (np.array, shape (batch, 1)): The answer(s) chosen for the control variant.
            treatment_answer (np.array, shape (batch, 1)): The answer(s) chosen for the treatment variant.
            answer_options (np.array, shape (batch, num_options)): The answer options available in the test case.
            anchor (np.array, shape (batch, 1)): The anchor value(s) used in the test case.

        Returns:
            The anchoring bias metric value.
        """

        if self.anchor_agnostic:
            if self.overall:
                return np.mean(
                    self.anchor_agnostic_metric(
                        control_answer, treatment_answer, answer_options
                    )
                )
            return self.anchor_agnostic_metric(
                control_answer, treatment_answer, answer_options
            )
        if self.overall:
            return np.mean(
                self.anchor_specific_metric(
                    control_answer, treatment_answer, answer_options, anchor
                )
            )
        return self.anchor_specific_metric(
            control_answer, treatment_answer, answer_options, anchor
        )


class HaloEffectMetric:
    """
    A class that describes the quantitative evaluation of the halo effect bias in a model.

    Individual metric:
    𝔅 = ‖ â₂ − â₁ ‖₁ / ‖ â₁ − ã₁ ‖₁ ∈ [0, 1];

    Batch metric:
    𝔅 = (𝔅₁ + ... + 𝔅ₙ) / n,

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    ã₁ is the farthest answer option from â₁;
    n is number of test cases in the batch.

    Attributes:
        overall (bool): A flag that is used to indicate that a single result per batch of test is required.

    """

    def __init__(self, overall: bool):
        self.overall = overall

    def compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        answer_options: np.array,
    ) -> np.array:
        """
        Computes the halo effect bias metric for the given batch of test instances.

        Args:
            control_answer (np.array, shape (batch, 1)): The answer(s) chosen for the control variant.
            treatment_answer (np.array, shape (batch, 1)): The answer(s) chosen for the treatment variant.
            answer_options (np.array, shape (batch, num_options)): The answer options available in the test case.

        Returns:
            The halo effect bias metric value.
        """
        farthest_idx = np.argmax(
            np.abs(answer_options - control_answer), axis=1, keepdims=True
        )
        farthest_val = np.take_along_axis(answer_options, farthest_idx, axis=1)
        result = np.abs(control_answer - treatment_answer) / np.abs(
            control_answer - farthest_val + 1e-8
        )
        if self.overall:
            return np.mean(result)

        return result


class LossAversionMetric:
    """
    A class that describes the quantitative evaluation of the loss aversion bias in a model.

    Individual metric:
    𝔅ᵢ = aᵢ ∀i = 1,.., n;


    Batch metric:
    𝔅 = 1 - (𝔅₁/λ₁ + ... + 𝔅ₙ/λₙ) / (1/λ₁ + ... + 1/λₙ) ∈ [0, 1],

    where:
    aᵢ ∈ {0,1} is the chosen answer for the i-th test;
    λᵢ is the loss aversion hyperparameter in the i-th test, decreased by 1 (for sharpness purpose).

    Attributes:
        overall (bool): A flag that is used to indicate that a single result per batch of test is required.
    """

    def __init__(self, overall: bool):
        self.overall = overall

    def compute(self, answer: np.array, lambda_val: np.array) -> np.array:
        """
        Computes the loss aversion bias metric for the given batch of test instances.

        Args:
            answer (np.array, shape (batch, 1)): The answer(s) chosen.
            lambda_val (np.array, shape (batch, 1)): The loss aversion hyperparameter(s).

        Returns:
            The loss aversion bias metric value.
        """
        if not self.overall:
            return answer

        lambda_val = lambda_val
        result = 1 - np.sum(answer / lambda_val) / np.sum(1 / lambda_val)

        return result


class ConfirmationBiasMetric:
    """
    A class that describes the quantitative evaluation of the confirmation bias in a model.

    Individual metric:
    𝔅ᵢ = max(0, [aᵢ(aᵢ⁺ − aᵢ⁻) + (1 − aᵢ)(aᵢ⁻ − aᵢ⁺)]/(aᵢ⁺ + aᵢ⁻))  ∀i = 1,.., n;


    Batch metric: [TODO: potentially can also use simple average]
    𝔅 = (𝔅₁N₁ + ... + 𝔅ₙNₙ) / (N₁ + ... + Nₙ),

    where:
    aᵢ ∈ {0,1} is the chosen answer in the control version of the i-th test;
    aᵢ⁺ is the number of pro-arguments selected in the treatment version of the i-th test;
    aᵢ⁻ is the number of con-arguments selected in the treatment version of the i-th test;
    Nᵢ is the number of arguments in the treatment version of the i-th test;
    n is number of test cases in the batch.

    Attributes:
        overall (bool): A flag that is used to indicate that a single result per batch of test is required.
    """

    def __init__(self, overall: bool):
        self.overall = overall

    def compute(
        self,
        answer: np.array,
        pro_answer: np.array,
        con_answer: np.array,
        n_args: np.array,
    ) -> np.array:
        """
        Computes the confirmation bias metric for the given batch of test instances.

        Args:
            answer (np.array, shape (batch, 1)): The answer(s) chosen in the control version(s).
            pro_answer (np.array, shape (batch, 1)): The number of pro-arguments chosen in the treatment version(s).
            con_answer (np.array, shape (batch, 1)): The number of con-arguments chosen in the treatment version(s).
            n_args (np.array, shape (batch, 1)): The number of arguments available in the treatment version(s).

        Returns:
            The confirmation bias metric value.
        """
        result = np.maximum(
            0,
            (
                answer * (pro_answer - con_answer)
                + (1 - answer) * (con_answer - pro_answer)
            )
            / (pro_answer + con_answer),
        )
        if not self.overall:
            return result

        result = np.sum(result * n_args) / np.sum(n_args)

        return result
