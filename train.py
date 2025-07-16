"""
Script d'entraînement flexible pour le modèle MIPANet.
Permet d'entraîner avec différentes combinaisons de data_types.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from tqdm import tqdm

# Spécification de la carte GPU à utiliser
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Importation des modules du projet
from model.model import get_mipanet
from model.datasets import get_dataset
from config import get_config
from model.net.loss import SegmentationLoss
import model.utils as utils
from model.datasets.transforms import transforms, irc_mean, irc_std

class Trainer:
    """
    Classe Trainer pour gérer l'entraînement avec différentes combinaisons de data_types.
    """
    def __init__(self, root, data_types):
        
        self.root = root
        self.data_types = data_types
        self.config = get_config(data_types = data_types)

        # Nom de l'entrainement basé sur les paramètres
        data_types_str = "+".join(data_types)
        self.NAME = f"runs-{self.config['training']['dataset']}-ep_{self.config['training']['epochs']}-bs_{self.config['training']['batch_size']}-lr_{self.config['training']['lr']}-{self.config['training']['commentaire']}"

        # Chemin par défaut pour les résultats
        self.RESULT_PATH = root / 'MIPANet/train' / self.NAME
        self.TB_LOG_PATH = root / 'MIPANet/train/tensorboard_logs'

        print(
            "Configuration d'entraînement :\n\n"
            f"  Data types     : {' + '.join(data_types)}\n"
            f"  Dataset        : {self.config['training']['dataset']}\n"
            f"  Batch size     : {self.config['training']['batch_size']}\n"
            f"  Epochs         : {self.config['training']['epochs']}\n"
            f"  Workers        : {self.config['training']['workers']}\n\n"
            f"  Learning rate  : {self.config['training']['lr']}\n"
            f"  LR scheduler   : {self.config['training']['lr_scheduler']}\n"
            f"  Momentum       : {self.config['training']['momentum']}\n\n"
            f"  First fusions  : {self.config['encoder']['first_fusions']}\n"
            f"  Last fusion    : {self.config['encoder']['last_fusion']}\n"
            f"  Use CUDA       : {self.config['training']['use_cuda']}\n"
        )

        # Chargement des datasets d'entraînement et de validation
        trainset = get_dataset(
            self.config['training']['dataset'],
            root=self.root,
            split='train',
            data_types=data_types
        )
        
        testset = get_dataset(
            self.config['training']['dataset'],
            root=self.root, 
            split='val',
            data_types=data_types
        )

        # Création des DataLoaders
        kwargs = {'num_workers': self.config['training']['workers'], 'pin_memory': True} if self.config['training']['use_cuda'] else {}
        self.trainloader = data.DataLoader(trainset, batch_size=self.config['training']['batch_size'], drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=self.config['training']['batch_size'], drop_last=False, shuffle=False, **kwargs)

        self.train_step = len(self.trainloader) // 4
        self.val_step = len(self.valloader) // 4
        self.nclass = trainset.num_class

        # Initialisation du modèle
        model = get_mipanet(
            dataset=self.config['training']['dataset'],
            data_types=data_types,
            pass_rff=self.config['encoder']['pass_rff'],
            first_fusions=self.config['encoder']['first_fusions'],
            last_fusion=self.config['encoder']['last_fusion'],
            use_tgcc=self.config['TGCC']['use_TGCC'],
        )

        # Configuration de l'optimiseur
        enc = model.encoder
        base_modules = []
        
        # Collecter les modules de base pour chaque branche
        for dt in data_types:
            base_modules.append(enc.branches[f'{dt}_base'])
        
        base_ids = utils.get_param_ids(base_modules)
        base_params = filter(lambda p: id(p) in base_ids, model.parameters())
        other_params = filter(lambda p: id(p) not in base_ids, model.parameters())
        
        self.optimizer = torch.optim.SGD([
            {'params': base_params, 'lr': self.config['training']['lr']},
            {'params': other_params, 'lr': self.config['training']['lr'] * 10}
        ], momentum=self.config['training']['momentum'],
           weight_decay=self.config['training']['weight_decay'])

        # Planificateur de taux d'apprentissage
        self.scheduler = utils.LR_Scheduler_Head(
            mode=self.config['training']['lr_scheduler'],
            base_lr=self.config['training']['lr'],
            num_epochs=self.config['training']['epochs'],
            iters_per_epoch=len(self.trainloader),
            warmup_epochs=5
        )
        self.best_pred = (0.0, 0.0)

        # Configuration du dispositif (CPU ou GPU)
        self.device = torch.device("cuda:0" if self.config['training']['use_cuda'] else "cpu")
        if self.config['training']['use_cuda']:
            if torch.cuda.device_count() > 1:
                GPUS = list(range(torch.cuda.device_count()))
                print("Utilisation de", torch.cuda.device_count(), "GPUs !")
                model = nn.DataParallel(model, device_ids=GPUS)
                self.multi_gpu = True
            else:
                self.multi_gpu = False
        else:
            self.multi_gpu = False

        self.model = model.to(self.device)

        # Définition de la fonction de perte
        self.criterion = SegmentationLoss(
            aux=True,
            aux_weight=self.config['training']['aux_weight'],
            nclass=self.nclass,
            weight=None
        )

        utils.mkdir(self.TB_LOG_PATH)
        self.writer = SummaryWriter(str(self.TB_LOG_PATH))

    def denormalize(self, tensor, mean, std):
        """Dénormalise un tenseur."""
        mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
        return tensor * std + mean
    
    def colorize_mask(self, mask):
        """Colorise un masque de segmentation."""
        mask_np = mask.cpu().numpy()
        class_colors = {
            0: (0, 0, 0),          # noir = sol
            1: (127, 127, 127),    # gris = foret
            2: (255, 255, 255),    # blanc = vieille foret
        }
        h, w = mask_np.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in class_colors.items():
            color_mask[mask_np == class_id] = color

        return color_mask

    def create_composite(self, data_inputs, pred, target, epoch, phase, composite_idx):
        """Crée un composite IRC-prédiction-target pour TensorBoard."""
        # Prendre le premier élément du batch
        irc_input = data_inputs[0][0]  # Premier data_type (IRC), premier échantillon
        pred_mask = torch.argmax(pred[0][0], dim=0)  # Prédiction, premier échantillon
        target_mask = target[0]  # Target, premier échantillon
        
        # Dénormaliser l'image IRC
        irc_denorm = self.denormalize(irc_input, irc_mean, irc_std)
        irc_denorm = torch.clamp(irc_denorm, 0, 1)
        
        # Coloriser les masques
        pred_colored = self.colorize_mask(pred_mask)
        target_colored = self.colorize_mask(target_mask)
        
        # Convertir IRC en numpy et redimensionner si nécessaire
        irc_np = irc_denorm.cpu().numpy().transpose(1, 2, 0)
        if irc_np.shape[2] == 3:
            irc_display = (irc_np * 255).astype(np.uint8)
        else:
            # Si c'est une image mono-canal, la répéter sur 3 canaux
            irc_display = np.repeat((irc_np[:, :, 0:1] * 255).astype(np.uint8), 3, axis=2)
        
        # Créer le composite horizontal
        composite = np.hstack([irc_display, pred_colored, target_colored])
        
        # Ajouter à TensorBoard avec un tag qui permet la navigation chronologique
        tag = f"Visualisation/{phase}_composite_{composite_idx}"
        self.writer.add_image(tag, composite, epoch, dataformats='HWC')

    def training(self, epoch):
        """Entraîne le modèle pour une époque donnée."""
        train_loss = 0.0
        self.model.train()

        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        
        # Variables pour capturer les composites
        composite_saved = 0
        composite_interval = len(self.trainloader) // 2  # Pour avoir 2 composites par époque
        
        for i, batch in tqdm(enumerate(self.trainloader), total=len(self.trainloader)):
            self.scheduler(self.optimizer, i, epoch, sum(self.best_pred))
            self.optimizer.zero_grad()

            # Séparer les données et le target
            *data_inputs, target, name = batch
            
            # Déplacer sur le device
            data_inputs = [x.to(self.device) for x in data_inputs]
            target = target.to(self.device)

            outputs = self.model(*data_inputs)
            loss = self.criterion(*outputs, target)

            loss.backward()
            self.optimizer.step()

            # Sauvegarder les composites à intervalles réguliers
            if composite_saved < 2 and i % composite_interval == 0:
                with torch.no_grad():
                    self.create_composite(data_inputs, outputs, target, epoch, 'train', composite_saved + 1)
                    composite_saved += 1

            correct, labeled = utils.batch_pix_accuracy(outputs[0].data, target)
            inter, union = utils.batch_intersection_union(outputs[0].data, target, self.nclass)
            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            train_loss += loss.item()

            if (i + 1) % self.train_step == 0:
                avg_loss = train_loss / self.train_step
                print('Époque {}, étape {}, perte {}'.format(epoch + 1, i + 1, avg_loss))
                self.writer.add_scalar('loss/train', avg_loss, epoch * len(self.trainloader) + i)
                train_loss = 0.0

        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IOU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIOU = IOU.mean()
        print('Époque {}, précision pixel {}, IOU moyen {}'.format(epoch + 1, pixAcc, mIOU))
        self.writer.add_scalar("metrics/mean_iou/train", mIOU, epoch)
        self.writer.add_scalar("metrics/pixel accuracy/train", pixAcc, epoch)

    def validation(self, epoch):
        """Évalue le modèle sur le jeu de validation."""
        def eval_batch(model, *data_inputs, target):
            pred = model(*data_inputs)
            loss = self.criterion(*pred, target)
            correct, labeled = utils.batch_pix_accuracy(pred[0].data, target)
            inter, union = utils.batch_intersection_union(pred[0].data, target, self.nclass)
            return correct, labeled, inter, union, loss, pred

        self.model.eval()
        total_inter, total_union, total_correct, total_label, total_loss = 0, 0, 0, 0, 0
        
        # Variables pour capturer les composites
        composite_saved = 0
        composite_interval = len(self.valloader) // 2  # Pour avoir 2 composites par époque
        
        for i, batch in enumerate(self.valloader):
            *data_inputs, target, name = batch
            data_inputs = [x.to(self.device) for x in data_inputs]
            target = target.to(self.device)

            with torch.no_grad():
                correct, labeled, inter, union, loss, pred = eval_batch(self.model, *data_inputs, target=target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            total_loss += loss.item()
            
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IOU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIOU = IOU.mean()

            # Sauvegarder les composites à intervalles réguliers
            if composite_saved < 2 and i % composite_interval == 0:
                with torch.no_grad():
                    self.create_composite(data_inputs, pred, target, epoch, 'val', composite_saved + 1)
                    composite_saved += 1

            if i % self.val_step == 0:
                print('Évaluation IOU moyen {}'.format(mIOU))

        loss = total_loss / len(self.valloader)
        self.writer.add_scalar("metrics/mean_iou/val", mIOU, epoch)
        self.writer.add_scalar("metrics/pixel accuracy/val", pixAcc, epoch)
        self.writer.add_scalar("loss/val", loss, epoch)

        return pixAcc, mIOU, loss

    def train_n_evaluate(self):
        """Entraîne et évalue le modèle sur plusieurs époques."""
        results = {'miou': [], 'pix_acc': []}

        for epoch in range(self.config['training']['epochs']):
            print(f"\n=============== Entraînement de l'époque {epoch + 1}/{self.config['training']['epochs']} ==========================")
            self.training(epoch)

            print(f'\n=============== Début de l\'évaluation, époque {epoch + 1} ===============\n')
            pixAcc, mIOU, loss = self.validation(epoch)
            print('Évaluation précision pixel {}, IOU moyen {}, perte {}'.format(pixAcc, mIOU, loss))

            results['miou'].append(round(mIOU, 6))
            results['pix_acc'].append(round(pixAcc, 6))

            # Sauvegarde du meilleur modèle
            is_best = False
            new_pred = (round(mIOU, 6), round(pixAcc, 6))
            if sum(new_pred) > sum(self.best_pred):
                is_best = True
                self.best_pred = new_pred
                best_state_dict = self.model.module.state_dict() if self.multi_gpu else self.model.state_dict()

            if is_best:
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict() if self.multi_gpu else self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred
                }, self.config, is_best)

        # Moyenne des 5 dernières époques
        final_miou = sum(results['miou'][-5:]) / 5
        final_pix_acc = sum(results['pix_acc'][-5:]) / 5

        final_result = '\nPerformance des 5 dernières époques\n[mIoU]: %4f\n[Pixel_Acc]: %4f\n[Meilleure Prédiction]: %s\n' % (
            final_miou, final_pix_acc, self.best_pred)
        print(final_result)

        # Exportation des poids si nécessaire
        flag = (final_miou > 0.1)
        if self.config['training']['export'] or flag:
            export_info = f"{'_'.join(self.data_types)}_{int(time.time())}"
            utils.mkdir(self.RESULT_PATH)
            torch.save(best_state_dict, self.RESULT_PATH / f"{export_info}.pth")
            print(f'Exporté sous {export_info}.pth')
        
        self.writer.close()
        print(f"Logs TensorBoard sauvegardés dans : {self.TB_LOG_PATH}/{export_info}.pth")

def train(root, data_types=['irc', 'mnh', 'biom']):
    """
    Fonction principale pour l'entraînement.
    
    Exemples d'utilisation:
    - train(root, data_types=['irc', 'mnh'])
    - train(root, data_types=['irc', 'biom'])
    - train(root, data_types=['irc', 'mnh', 'histo'])
    """
    start_time = time.time()
    print(f"\n------- Début de l'entraînement avec {' + '.join(data_types)} ----------\n")

    trainer = Trainer(
        root=root,
        data_types=data_types
        )
    trainer.train_n_evaluate()

    # Afficher la durée de l'entrainement
    elapsed_secs = int(time.time() - start_time)
    hours, remainder = divmod(elapsed_secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"[Temps écoulé] : {hours}h {minutes}min {seconds}s")