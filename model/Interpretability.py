import os
import matplotlib.pyplot as plt
import torch.nn.functional as F


def visualize_attention_weights(model, dataloader, device, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model.eval()
    img_batch, _ = next(iter(dataloader))
    img_batch = img_batch.to(device)


    attention_weights = model.get_attention_weights(img_batch)


    attention_weights = attention_weights.mean(dim=1)[:, 0, 1:]


    num_patches_side = int((attention_weights.size(-1) + 1) ** 0.5)
    attention_map = attention_weights.view(-1, num_patches_side, num_patches_side)
    img_to_visualize = img_batch[0]
    attention_map_to_visualize = attention_map[0]
    attention_map_upsampled = F.interpolate(
        attention_map_to_visualize.unsqueeze(0).unsqueeze(0),
        size=img_to_visualize.shape[-2:],
        mode='bilinear',
        align_corners=False
    ).squeeze(0).detach()

    img_to_visualize = img_to_visualize.cpu().numpy().transpose(1, 2, 0)
    attention_map_upsampled = attention_map_upsampled.squeeze(0).cpu().numpy()


    img_to_visualize = (img_to_visualize - img_to_visualize.min()) / (img_to_visualize.max() - img_to_visualize.min())
    attention_map_upsampled = (attention_map_upsampled - attention_map_upsampled.min()) / (
                attention_map_upsampled.max() - attention_map_upsampled.min())


    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_to_visualize)
    ax.imshow(attention_map_upsampled, cmap='jet', alpha=0.5)
    ax.set_title('Attention Map Overlay')
    plt.axis('off')


    output_path = os.path.join(output_folder, 'attention_map.png')
    plt.savefig(output_path)
    plt.close(fig)