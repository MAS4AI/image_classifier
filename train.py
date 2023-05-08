import argparse

'''to run the code
 > python train.py /home/workspace/ImageClassifier/flowers --gpu --epochs 8
'''
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict

def create_model(arch, hidden_units):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_features = model.classifier[0].in_features
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_features = model.classifier[0].in_features
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_features = model.classifier[0].in_features
    else:
        raise ValueError("Please select vgg16, vgg19, or vgg13.")

    # Modify the classifier with the new input features
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model


def train_model(data_dir, arch, hidden_units, learning_rate, epochs, gpu ,save_directory):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    model = create_model(arch, hidden_units)
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    print("Training started...")
    steps = 0
    running_loss = 0
    print_every = 40
    dirr = save_directory
    print(f'this is were you are saving the model :> {dirr}')
    for epoch in range(epochs):
        
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model(inputs)

                        batch_loss = criterion(output, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                train_loss=running_loss/print_every
                valid_loss=valid_loss/len(valid_loader)
                accuracy= accuracy/len(valid_loader)                          
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Step {steps}.. "
                      f"Train loss: {train_loss:.3f}.. "
                      f"Validation loss: {valid_loss:.3f}.. "
                      f"Validation accuracy: {accuracy*100:.3f}")

                running_loss = 0
                model.train()

    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'architecture': arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    
    torch.save(checkpoint, dirr+'/checkpoint.pth')

    print("Training completed and model saved.")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a deep neural network on a dataset')
    parser.add_argument('data_directory', type=str, help='path to the dataset')
    parser.add_argument('--save_directory', type=str, default='/home/workspace/ImageClassifier', help='directory to save the trained model')
    parser.add_argument('--arch', type=str, default='vgg16', help='architecture of the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=4096, help='number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training')
    args = parser.parse_args()

 
    train_model(args.data_directory, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu,args.save_directory)


if __name__ == '__main__':
    main()

                        