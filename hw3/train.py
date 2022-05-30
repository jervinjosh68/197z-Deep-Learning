import torch
import kws_data
import transformer
import os
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == '__main__':
    args = transformer.get_args()
    CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
               'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
    if not os.path.exists('./dataset'):
        print("Dataset folder is not found. Downloading...")
        data_path = "./dataset"
        download = True
        os.makedirs(data_path)
    
    else:
        data_path = "./dataset"
        download = False
        print("Dataset folder is found. Loading data...")
    datamodule = kws_data.KWSDataModule(batch_size=args.batch_size, num_workers=args.num_workers,
                               path="dataset", n_fft=args.n_fft, n_mels=args.n_mels,
                               win_length=args.win_length, hop_length=args.hop_length,
                               class_dict=CLASS_TO_IDX, patch_num=args.patch_num,download=download)
    datamodule.setup()

    data = iter(datamodule.train_dataloader()).next()
    patch_dim = data[0].shape[-1]
    seqlen = data[0].shape[-2]
    model = transformer.LitTransformer(num_classes=args.num_classes, lr=args.lr, epochs=args.max_epochs, 
                           depth=args.depth, embed_dim=args.embed_dim, head=args.num_heads,
                           patch_dim=patch_dim, seqlen=seqlen, patch_num = args.patch_num)

    if not os.path.exists("./checkpoints"):
        print("Checkpoints folder is not found. Creating...")
        checkpoint_path = "./checkpoints"
        os.makedirs(checkpoint_path)
    else:
        checkpoint_path = "./checkpoints"
        print("Checkpoints folder is found.")
    model_checkpoint = ModelCheckpoint(
        dirpath="checkpoints",
        filename="pl-transformer-kws-best-acc",
        save_top_k=1,
        verbose=True,
        monitor='test_acc',
        mode='max',
    )
    trainer = transformer.Trainer(accelerator=args.accelerator, devices=args.devices,
                      max_epochs=args.max_epochs, precision=16 if args.accelerator == 'gpu' else 32,)
    model.hparams.sample_rate = datamodule.sample_rate
    model.hparams.idx_to_class = idx_to_class
    trainer.fit(model, datamodule=datamodule)
    #model = model.load_from_checkpoint(os.path.join("checkpoints", "pl-kws-best-acc.ckpt"))
    model.eval()
    script = model.to_torchscript()

    # save for use in production environment
    model_path = os.path.join("checkpoints",
                          "pl-transformer-kws-best-acc.pt")
    torch.jit.save(script, model_path)






    

    