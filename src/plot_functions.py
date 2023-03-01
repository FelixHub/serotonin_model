import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 300
import numpy as np


def plot_video_comparison(video, reconstructed_video):
    nb_frame = video.shape[0]
    fig, ax = plt.subplots(2, nb_frame, figsize=(nb_frame, 2))
    for i in range(video.shape[0]):
        ax[0, i].imshow(video[i, :, :])
        ax[0, i].axis("off")
    for i in range(video.shape[0]):
        ax[1, i].imshow(reconstructed_video[i, :, :])
        ax[1, i].axis("off")
    fig.tight_layout()


def plot_SPE_metrics(latent_video, latent_video_variance, mean_pred, sig_pred):
    z = np.mean(latent_video, axis=0).T
    z_pred = np.mean(mean_pred, axis=0).T

    # first measure of SPE : norm between z_t and mean of P(z_t|h_t-1)
    z_error = np.abs(z_pred[:, :-1] - z[:, 1:])
    z_error_norm = np.linalg.norm(z_error, axis=0)

    # second measure of SPE : variance of P(z_t|h_t-1) (more like predicted uncertainty)
    var_pred = np.mean(sig_pred, axis=0).T
    var_pred_avg = np.mean(np.mean(sig_pred, axis=0).T, axis=0)

    # third measure of SPE : KL divergence between P(z_t|x_t) and P(z_t|h_t-1)
    z_pred_mean = mean_pred[:, :-1, :]
    z_pred_var = sig_pred[:, :-1, :]
    z_mean = latent_video[:, 1:, :]
    z_var = latent_video_variance[:, 1:, :]
    kl_loss = -0.5 * np.sum(
        np.expand_dims(z_var, -1)
        - np.expand_dims(z_pred_var, -1)
        - np.divide(
            (
                np.expand_dims(z_var, -1)
                + (np.expand_dims(z_mean, -1) - np.expand_dims(z_pred_mean, -1)) ** 2
            ),
            np.expand_dims(z_pred_var, -1) + 1e-10,
        ),
        axis=-1,
    )
    kl_loss = np.mean(kl_loss, axis=0).T
    kl_loss_avg = np.mean(kl_loss, axis=0)

    f = plt.figure(figsize=(5, 10))

    plt.subplot(421)
    plt.imshow(z)
    plt.colorbar()
    plt.xlabel("frame")
    plt.ylabel(r"latent state $z_{t}$")

    plt.subplot(422)
    plt.imshow(z_pred)
    plt.colorbar()
    plt.xlabel("frame")
    plt.ylabel(r"prediction of $z_{t+1}$ ")

    plt.subplot(423)
    plt.imshow(z_error)
    plt.colorbar()
    plt.xlabel("frame")
    plt.ylabel("latent prediction error")

    plt.subplot(424)
    plt.plot(z_error_norm)
    plt.xlabel("video frame")
    plt.title(r"$||z_{t+1} - ẑ_{t+1}||$")

    plt.subplot(425)
    plt.imshow(var_pred)
    plt.colorbar()
    plt.xlabel("frame")
    plt.ylabel("prediction variance")

    plt.subplot(426)
    plt.plot(var_pred_avg)
    plt.xlabel("video frame")
    plt.title("mean latent prediction variance")

    plt.subplot(427)
    plt.imshow(kl_loss)
    plt.colorbar()
    plt.xlabel("frame")
    plt.ylabel("KL-div between pred. and true latent")

    plt.subplot(428)
    plt.plot(kl_loss_avg)
    plt.xlabel("frame")
    plt.title("mean KL-divergence")

    f.suptitle("measures of state prediction error")
    f.tight_layout()
    plt.show()
