import pytorch_lightning as pl


class Agent(pl.LightningModule):

	def __init__(self):

		super(Agent, self).__init__()

	def forward(self, x):
		pass

	def training_step(self, batch, batch_idx):
		x, y = batch
		pass

	def training_step_end(self, batch_parts):
		# train network?
		# run env?
		pass

	def training_epoch_end(self, training_step_outputs):
		pass

	def validation_step(self, batch, batch_idx):
		# check elo score
		pass

	def validation_step_end(self, batch_parts):
		pass

	def validation_epoch_end(self, training_step_outputs):
		pass
	   