import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import ceil
from pathlib import Path


def custom_scatter(
    df,
    response,
    predictors,
    axis_labels=None,
    ncols=3,
    figsize_per_col=4.5,
    figsize_per_row=4.5,
    title=None,
    sharey=True,
    scatter_kws=None,
):
    # Handle case where axis_labels dict or scatter_kws is not provided
    if axis_labels == None:
        axis_labels = {}
    if scatter_kws == None:
        scatter_kws = {"alpha": 0.35, "s": 20, "edgecolors": "none"}

    # Set number of columns and rows in the figure
    n = len(predictors)
    nrows = ceil(n / ncols)

    # Create plots
    fig, axes = plt.subplots(nrows, ncols, figsize=(
        figsize_per_col * ncols, figsize_per_row * nrows), sharey=sharey)
    axes = np.array(axes).reshape(-1)

    for i, pred in enumerate(predictors):
        ax = axes[i]
        ax.scatter(df[pred], df[response], **scatter_kws)
        ax.set_xlabel(axis_labels.get(pred, pred))

        # Only label y-axis on left column to reduce clutter
        if (i % ncols) == 0:
            ax.set_ylabel(axis_labels.get(response, response))
        else:
            ax.set_ylabel("")

        ax.set_title(f"{response} vs. {pred}", fontsize=12)
        ax.grid(True, linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Add a title if wanted
    if title:
        fig.suptitle(title, fontsize=18, y=0.96)

    # Make plots fit nicely in the figure
    fig.subplots_adjust(hspace=0.35, wspace=0.30, top=0.90)

    return fig, axes


def custom_ccpr_plot(
    response,
    predictors,
    model,
    axis_labels=None,
    ncols=3,
    figsize_per_col=4.5,
    figsize_per_row=4.5,
    title=None,
    sharey=True,
    plot_kw=None,
    plot_titles=True,
):
    n = len(predictors)
    nrows = ceil(n / ncols)

    if axis_labels == None:
        axis_labels = {}
    if plot_kw == None:
        plot_kw = {"alpha": 0.35, "s": 14, "edgecolors": "none"}

    fig, axes = plt.subplots(nrows, ncols, figsize=(
        figsize_per_col*ncols, figsize_per_row*nrows), sharey=sharey)
    axes = np.array(axes).reshape(-1)

    for i, pred in enumerate(predictors):
        ax = axes[i]
        sm.graphics.plot_ccpr(model, pred, ax=ax)

        # Style the ccpr plot like my other custom plots
        alpha = plot_kw.get("alpha", 0.35)
        s = plot_kw.get("s", 14)
        edge = plot_kw.get("edgecolors", "none")

        for line in ax.lines:
            line.set_alpha(alpha)
            line.set_markersize((s ** 0.5))
            if edge == "none":
                line.set_markeredgewidth(0)

        # axis labels
        ax.set_xlabel("Fitted Value")

        # Only label y-axis on left column to reduce clutter
        if (i % ncols) == 0:
            ax.set_ylabel("Residual")
        else:
            ax.set_ylabel("")

        if plot_titles == True:
            if title:
                ax.set_title(f"{pred}", fontsize=12)
            else:
                ax.set_title(f"Partial Residual Plot for {pred}", fontsize=12)
        else:
            ax.set_title("", fontsize=12)
        ax.grid(True, linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Add a title if wanted
    if title:
        fig.suptitle(title, fontsize=18, y=0.96)

    # Make plots fit nicely in the figure
    fig.subplots_adjust(hspace=0.35, wspace=0.30, top=0.90)

    return fig, axes


def custom_qq_plot(residuals, figsize=(4.5, 4.5), title=None, line="45"):
    marker_kws = {"alpha": 0.6, "markersize": 4}
    fig, ax = plt.subplots(figsize=figsize)

    sm.qqplot(residuals, line=line, ax=ax, fit=True, marker="o", **marker_kws)

    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title("Normal Q-Q Plot of Residuals", fontsize=14)

    # Match styling
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig, ax


def save_fig(fig, filename, folder="figures", dpi=300, tight=True):
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = Path(folder) / filename
    fig.savefig(path, dpi=dpi)
    return str(path)
