import os
import torch
from model import Generator
from util import get_test_conditions, save_image
from evaluator import EvaluationModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 100
c_dim = 200
G_times = 4
test_path = os.path.join('data', 'test.json')
generator_path = os.path.join(
	'models', f'c_dim {c_dim} G{G_times}', 'epoch189_score0.72.pt')

if __name__ == '__main__':
	# load testing data conditions
	conditions1 = get_test_conditions(os.path.join('data', 'test.json')).to(device)  # (N,24) tensor
	conditions2 = get_test_conditions(os.path.join('data', 'new_test.json')).to(device)

	# load generator model
	g_model = Generator(z_dim, c_dim).to(device)
	g_model.load_state_dict(torch.load(generator_path))

	# test
	z = torch.randn(len(conditions1), z_dim).to(device)  # (N,100) tensor
	gen_imgs = g_model(z, conditions1)
	evaluation_model = EvaluationModel()
	score1 = evaluation_model.eval(gen_imgs, conditions1)
	print(f'Test score: {score1:.2f}')
	save_image(gen_imgs, 'test.png', nrow=8, normalize=True)

	z = torch.randn(len(conditions2), z_dim).to(device)  # (N,100) tensor
	gen_imgs = g_model(z, conditions2)
	evaluation_model = EvaluationModel()
	score1 = evaluation_model.eval(gen_imgs, conditions2)
	print(f'New test score: {score1:.6f}')
	save_image(gen_imgs, 'new test.png', nrow=8, normalize=True)