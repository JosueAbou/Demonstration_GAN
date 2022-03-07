import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter  #Afin de pouvoir écrire sur le tensorboard


#Création du discriminateur
class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

#Création du générateur
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  
        )

    def forward(self, x):
        return self.gen(x)


""" Paramétrisation du système """
device = "cpu"

#Learning rate,  Default_value=1e-4
lr = 3e-4

#Taille du vecteur aléatoire
z_dim = 64

#Dimensions de l'image : 784
image_dim = 28 * 28 * 1  

batch_size = 32
num_epochs = 50

#Instanciation du générateur et du générateur
discriminator = Discriminator(image_dim).to(device)
generator = Generator(z_dim, image_dim).to(device)

#Fonction d'optimisation qui sera utilisé pour mettre à jour les poids lors de la rétropropagation
optimise_discriminator = optim.Adam(discriminator.parameters(), lr=lr)
optimise_generator = optim.Adam(generator.parameters(), lr=lr)

#Loss Function : Binary Cross Entropy
loss_function = nn.BCELoss()

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

#Téléchargement et Chargement du Jeu de données
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):

        #Entrainement du Discriminateur
        real = real.view(-1, 784).to(device)
        disc_real = discriminator(real).view(-1) 
        lossD_real = loss_function(disc_real, torch.ones_like(disc_real))
        
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = generator(noise)
        disc_fake = discriminator(fake).view(-1)
        lossD_fake = loss_function(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2

        discriminator.zero_grad()
        lossD.backward(retain_graph=True)
        optimise_discriminator.step()


        #Entrainement du Générateur
        output = discriminator(fake).view(-1)
        lossG = loss_function(output, torch.ones_like(output))

        generator.zero_grad() 
        lossG.backward()
        optimise_generator.step()


        #Affichage sur le TensorBoard
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )
            with torch.no_grad():
                fake = generator(noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1







##Pour améliorer les perfomances : 
# Utiliser les CNN
# Utiliser des réseaux de neuronnes un peu plus grand
#  








# Hyperparameters etc.
#device = "cuda" if torch.cuda.is_available() else "cpu"


#fake = gen(fixed_noise).reshape(-1, 1, 28, 28)

#fixed_noise = torch.randn((batch_size, z_dim)).to(device)


# normalize inputs to [-1, 1] so make outputs [-1, 1]