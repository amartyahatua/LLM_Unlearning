from trl import SFTTrainer

class GradientAscentSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=True, **kwargs):
        # Modify the loss function here
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = -outputs.loss
        return loss

class ScaledGradientAscentTrainer(SFTTrainer):
    def __init__(self, *args, scaling_factor=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaling_factor = scaling_factor

    def compute_loss(self, model, inputs, return_outputs=True, **kwargs):
        outputs = model(**inputs)
        loss = -self.scaling_factor * outputs.loss
        return loss


class WeightedUnlearningTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=True, **kwargs):
        outputs = model(**inputs)
        # Apply different weights to different types of forget data
        forget_weight = 1.5  # Adjust based on data type
        loss = -forget_weight * outputs.loss
        return loss
