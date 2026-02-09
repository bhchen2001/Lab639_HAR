from datetime import datetime
from tensorboardX import SummaryWriter
import os
import torch
import random
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config.config_lab639 import parse_args, Lab639Config
from trainer import Lab639Trainer

def train_classification(config):
    config.output_name = f"{config.output_name}_{datetime.today().strftime('%d-%m-%y_%H%M')}"
    print(f"Training with output name: {config.output_name}")

    writer = SummaryWriter(os.path.join(config.result_path, 'logs', config.output_name))
    layout = {
        'Multi-label Classification': {
            'class_val_accuracy': ['Multiline', ['class_val_accuracy/class{}'.format(i) for i in range(config.num_classes)]],
        },
    }
    writer.add_custom_scalars(layout)

    for arg in vars(config):
        writer.add_text(arg, str(getattr(config, arg)))
    
    train_obj = Lab639Trainer(config)
    train_obj.train(writer)

    return

def train_cross_validation(config):
    for fold in range(1, config.fold_num + 1):
        config.output_name = f"{config.exp_name}_fold{fold}"
        config.csv_offset = f"_fold{fold}"
        train_classification(config)

def test_classification(config, pth_path):

    test_obj = Lab639Trainer(config)
    return test_obj.test(pth_path)


def test_cross_validation(config):
    loss_list = []
    acc_list = []
    
    all_preds = []
    all_labels = []
    
    pth_list = os.listdir(os.path.join(config.model_path, 'pth'))
    for fold in range(1, config.fold_num + 1):
        pth_path = next((pth for pth in pth_list if f"fold{fold}" in pth), None)
        pth_path = os.path.join(config.model_path, 'pth', pth_path)
        # find files with .pth extension in pth_path
        pth_files = [f for f in os.listdir(pth_path) if f.endswith('.pth')]
        if not pth_files:
            raise ValueError(f"No .pth files found in {pth_path}. Please check the directory.")
        pth_path = os.path.join(pth_path, pth_files[0])

        config.csv_offset = f"_fold{fold}"

        print(f"Testing fold {fold} with model path: {pth_path}")
        loss, acc, fold_preds, fold_labels, miss_keys, miss_pred = test_classification(config, pth_path)

        loss_list.append(loss)
        acc_list.append(acc)

        all_preds.append(fold_preds)
        all_labels.append(fold_labels)

        # write all miss_keys to csv file
        if miss_keys:
            with open(os.path.join(config.model_path, 'miss_keys.csv'), 'a') as f:
                for i in range(len(miss_keys)):
                    f.write(f"{miss_keys[i]},{miss_pred[i]}\n")
            print(f"Miss keys saved to {os.path.join(config.model_path, 'miss_keys.csv')}")

    print(f"Average loss for all folds: {sum(loss_list) / len(loss_list)}")
    print(f"Average accuracy for all folds: {sum(acc_list) / len(acc_list)}")

    draw_confusion_matrix(all_preds, all_labels, config.num_classes, config.result_path, config.model_path, config.fusion_type)

def draw_confusion_matrix(all_preds, all_labels, num_classes, result_path, model_path, fusion_type):
    from sklearn.metrics import confusion_matrix
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    # import seaborn as sns

    action_list_y = ["drop", "walk", "sit", "stand up", "donning", "doffing", "throw", "carry", "pickup"]
    action_list_x = ["A{}".format(str(i).zfill(3)) for i in range(1, 11)]

    # confusion matrix for each fold
    for i, (preds, labels) in enumerate(zip(all_preds, all_labels)):
        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
        # Adjust figure size to fit axis labels
        plt.figure(figsize=(10, 8))  # Adjust width and height as needed

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=action_list_y)
        disp.plot(cmap=plt.cm.Blues, ax=plt.gca())  # Use the current axis for plotting

        # Set custom x-axis labels
        plt.gca().set_xticks(range(len(action_list_y)))
        plt.gca().set_xticklabels(action_list_y, rotation=45, ha="right")

        # Set custom y-axis labels
        plt.gca().set_yticks(range(len(action_list_y)))
        plt.gca().set_yticklabels(action_list_y)

        plt.title(f'Confusion Matrix - Fold {i + 1}')
        plt.tight_layout()  # Automatically adjust layout to fit labels
        plt.savefig((os.path.join(model_path, model_path.split('/')[-1] + f'_confusion_matrix_{i + 1}.png')))
        plt.close()

    print("Folds' Confusion matrices saved.")

    fsz = 20

    # Save confusion matrix for all folds
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    # Adjust figure size to fit axis labels
    plt.figure(figsize=(10, 8))  # Adjust width and height as needed

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=action_list_y)
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())  # Use the current axis for plotting

    ax = plt.gca()
    for text in plt.gca().texts:
        text.set_fontsize(fsz)

    # Set custom x-axis labels
    plt.gca().set_xticks(range(len(action_list_y)))
    plt.gca().set_xticklabels(action_list_y, rotation=45, ha="right", fontsize = fsz)

    # Set custom y-axis labels
    plt.gca().set_yticks(range(len(action_list_y)))
    plt.gca().set_yticklabels(action_list_y, fontsize = fsz)

    ax.set_xlabel("Predicted label", fontsize=fsz)
    ax.set_ylabel("True label", fontsize=fsz)

    plt.title(f'Confusion Matrix - All - {fusion_type} fusion')
    plt.tight_layout()  # Automatically adjust layout to fit labels
    plt.savefig(os.path.join(model_path, model_path.split('/')[-1] + '_confusion_matrix_all.png'))
    plt.close()
    print("All folds' Confusion matrix saved.")

def set_seed(config):
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    config = Lab639Config(args)
    set_seed(config)
    print(config)

    if config.mode == 'train':
        config.result_path = os.path.join(config.result_path, config.exp_name + '_' + datetime.today().strftime('%d-%m-%y_%H%M'))
        if not os.path.exists(config.result_path):
            os.makedirs(config.result_path)

        if config.fold_num > 1:
            train_cross_validation(config)
        else:
            train_classification(config)
        print("Training completed.")
    elif config.mode == 'test':
        config.model_path = os.path.join(config.result_path, config.model_path)
        if not os.path.exists(config.model_path):
            raise ValueError(f"Model path {config.model_path} does not exist.")
        
        # list folders in model path + 'pth'
        pth_list = os.path.join(config.model_path, 'pth')
        # if len(os.listdir(pth_list)) != config.fold_num:
        #     raise ValueError(f"Number of folders in {pth_list} is {len(os.listdir(pth_list))}, but config fold_num is {config.fold_num}")

        test_cross_validation(config)


if __name__ == "__main__":
    main()