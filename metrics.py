import numpy as np


class AnchoringBiasMetric:
    """
    A class that describes the quantitative evaluation of the anchoring bias in a model.
    Currently, two variations metrics are implemented: anchor agnostic and anchor specific,

    Agnostic: 𝔅 = ‖ â₂ − â₁ ‖₁ / ‖ â₁ − ã₁ ‖₁ ∈ [0,1];

    Specific: 𝔅' = max[0, (‖ â₁ − a' ‖₁ − ‖ â₂ − a' ‖₁) / ‖ â₁ − a' ‖₁] ∈ [0,1],

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    ã₁ is the farthest answer option from â₁;
    a' is the answer option closest to the anchor value.

    Attributes:
        anchor_agnostic (bool): A flag that is used for choice between two variations of the metric.

    """

    def __init__(self, anchor_agnostic: bool):
        self.anchor_agnostic = anchor_agnostic

    def anchor_agnostic_metric(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        answer_options: np.array,
    ) -> np.array:
        """
        Computes the anchor agnostic anchoring bias metric for the given control and treatment answers.

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
        Computes the anchor specific anchoring bias metric for the given control and treatment answers.

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
            return self.anchor_agnostic_metric(
                control_answer, treatment_answer, answer_options, anchor
            )
        else:
            return self.anchor_specific_metric(
                control_answer, treatment_answer, answer_options, anchor
            )


class HaloEffectMetric:
    """
    A class that describes the quantitative evaluation of the halo effect bias in a model.

    Metric:
    𝔅 = ‖ â₂ − â₁ ‖₁ / ‖ â₁ − ã₁ ‖₁ ∈ [0,1];

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    ã₁ is the farthest answer option from â₁.
    """

    def compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        answer_options: np.array,
    ) -> np.array:
        """
        Computes the halo effect bias metric for the given control and treatment answers.

        Args:
            control_answer (np.array): The answer(s) chosen for the control variant.
            treatment_answer (np.array): The answer(s) chosen for the treatment variant.
            answer_options (np.array): The answer options available in the test case.

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

        return result
