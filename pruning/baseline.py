from torchvision import models

if __name__ == "__main__":
    pretrain_model = models.alexnet(pretrained=True)

    test_data = ????
    test_label = ????

    pred = pretrain_model(test_data)


    total_acc = match(pred, test_label)
    total_parameter = get_param(pretrain_model)
    total_hyperparameter = get_hyper_param(pretrain_model)