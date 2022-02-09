import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

import numpy as np
from tqdm import tqdm

from model.vae import VAE
from dataset import TanksDataset, ToTensor



def train(vae, dataloader, epochs=1, device=torch.device("cpu")):
        vae = vae.to(device)
        vae = vae.double()
        #transform = T.ConvertImageDtype(dtype=torch.double)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
        reported_loss = []
        for epoch in range(epochs):

            collective_loss = []
            for _, x in tqdm(enumerate(dataloader)):

                x.to(device)
                
                #x = transform(images)

                #assert x.dtype == torch.double

                _, mu, log_sigma, x_prime = vae.forward(x.double())

                loss, recon, kld = vae.loss_fn(x, x_prime, mu, log_sigma)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                collective_loss.append([recon.item(), kld.item()])
            
            np_collective_loss = np.array(collective_loss)
            
            average_loss = np.mean(np_collective_loss, axis=1)

            reported_loss.append(average_loss)

            print(f"Epoch {epoch+1} finished!", f"reconstruction_loss = {average_loss[0]} || KL-Divergence = {average_loss[1]}", sep="\n")

            if (epoch+1) % 10 == 0:

                with torch.no_grad():
                    
                    to_img = T.ToPILImage()
                    
                    example = vae.sample()
                    
                    img_example = to_img(example)

                    img_example.save(f"result_at_epoch_{epoch+1}.png")
                    
        
        print("Training Finished!")

        return np.array(list(zip(range(epochs), average_loss)))



if __name__ == "__main__":
    

    train_loader = DataLoader(TanksDataset(transform=ToTensor()), batch_size=64, shuffle=True)

    vae = VAE(
        input_shape=[3, 64, 64],
        conv_filters=[3, 32, 64, 128 , 256],
        conv_kernels=[(5, 5), (3, 3), (3, 3), (3, 3)],
        conv_strides=[(1, 1), (1, 1), (1, 1), (1, 1)],
        paddings=[(1, 1), (1, 1), (1, 1), (1, 1)],
        output_paddings=[(0, 0), (0, 0), (0, 0), (0, 0)],
        dilations=[(1, 1), (1, 1), (1, 1), (1, 1)],
        latent_space_dim=1024
        )


    train(vae, train_loader, epochs=100, device=torch.device("cuda"))




