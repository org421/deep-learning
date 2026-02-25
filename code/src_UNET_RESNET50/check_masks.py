"""
Script pour vérifier les masks vides dans un dataset.

Usage:
    python check_masks.py --masks_dir path/to/masks
    python check_masks.py --masks_dir path/to/masks --images_dir path/to/images
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def check_masks(masks_dir: str, images_dir: str = None):
    """Vérifie les masks et affiche les statistiques."""
    
    masks_path = Path(masks_dir)
    
    if not masks_path.exists():
        print(f"❌ Dossier non trouvé: {masks_dir}")
        return
    
    # Extensions supportées
    mask_extensions = {'.npy', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    
    # Trouver tous les masks
    mask_files = [f for f in masks_path.iterdir() if f.suffix.lower() in mask_extensions]
    
    print(f"\n{'='*60}")
    print(f"VÉRIFICATION DES MASKS")
    print(f"{'='*60}")
    print(f"Dossier: {masks_dir}")
    print(f"Nombre de fichiers trouvés: {len(mask_files)}")
    print(f"{'='*60}\n")
    
    empty_masks = []
    valid_masks = []
    error_masks = []
    
    total_pixels_forged = 0
    
    for mask_file in sorted(mask_files):
        try:
            # Charger le mask
            if mask_file.suffix.lower() == '.npy':
                mask = np.load(mask_file)
            else:
                mask = np.array(Image.open(mask_file))
            
            # Gérer les masks 3D
            if len(mask.shape) == 3:
                if mask.shape[0] == 1:
                    mask = mask[0]
                else:
                    mask = mask[:, :, 0]
            
            # Binariser
            mask_binary = (mask > 0.5).astype(np.uint8)
            
            # Statistiques
            num_forged_pixels = mask_binary.sum()
            total_pixels = mask_binary.size
            percent_forged = 100 * num_forged_pixels / total_pixels
            
            if num_forged_pixels == 0:
                empty_masks.append(mask_file.name)
                status = "❌ VIDE"
            else:
                valid_masks.append(mask_file.name)
                total_pixels_forged += num_forged_pixels
                status = "✅"
            
            print(f"{status} {mask_file.name}: {num_forged_pixels:,} pixels forgés ({percent_forged:.2f}%) - shape: {mask.shape}")
            
        except Exception as e:
            error_masks.append((mask_file.name, str(e)))
            print(f"⚠️  {mask_file.name}: ERREUR - {e}")
    
    # Résumé
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ")
    print(f"{'='*60}")
    print(f"Total masks:     {len(mask_files)}")
    print(f"Masks valides:   {len(valid_masks)} ✅")
    print(f"Masks vides:     {len(empty_masks)} ❌")
    print(f"Masks en erreur: {len(error_masks)} ⚠️")
    print(f"{'='*60}")
    
    if empty_masks:
        print(f"\n⚠️  MASKS VIDES ({len(empty_masks)}):")
        for name in empty_masks:
            print(f"   - {name}")
    
    if error_masks:
        print(f"\n⚠️  MASKS EN ERREUR ({len(error_masks)}):")
        for name, err in error_masks:
            print(f"   - {name}: {err}")
    
    # Vérifier correspondance avec images si fourni
    if images_dir:
        images_path = Path(images_dir)
        if images_path.exists():
            print(f"\n{'='*60}")
            print(f"VÉRIFICATION CORRESPONDANCE IMAGES/MASKS")
            print(f"{'='*60}")
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
            
            # Chercher dans les sous-dossiers forged/authentic
            forged_dir = images_path / "forged"
            if forged_dir.exists():
                image_files = list(forged_dir.iterdir())
            else:
                image_files = list(images_path.iterdir())
            
            image_files = [f for f in image_files if f.suffix.lower() in image_extensions]
            image_stems = {f.stem for f in image_files}
            mask_stems = {f.stem.replace('_mask', '') for f in mask_files}
            
            images_without_mask = image_stems - mask_stems
            masks_without_image = mask_stems - image_stems
            
            print(f"Images trouvées: {len(image_files)}")
            print(f"Masks trouvés:   {len(mask_files)}")
            
            if images_without_mask:
                print(f"\n⚠️  Images SANS mask ({len(images_without_mask)}):")
                for name in sorted(images_without_mask)[:10]:
                    print(f"   - {name}")
                if len(images_without_mask) > 10:
                    print(f"   ... et {len(images_without_mask) - 10} autres")
            
            if masks_without_image:
                print(f"\n⚠️  Masks SANS image ({len(masks_without_image)}):")
                for name in sorted(masks_without_image)[:10]:
                    print(f"   - {name}")
                if len(masks_without_image) > 10:
                    print(f"   ... et {len(masks_without_image) - 10} autres")
            
            if not images_without_mask and not masks_without_image:
                print("✅ Correspondance parfaite images/masks")
    
    print(f"\n{'='*60}")
    
    return {
        'total': len(mask_files),
        'valid': len(valid_masks),
        'empty': len(empty_masks),
        'errors': len(error_masks),
        'empty_list': empty_masks,
    }


def main():
    parser = argparse.ArgumentParser(description='Vérifier les masks vides dans un dataset')
    parser.add_argument('--masks_dir', type=str, required=True,
                        help='Chemin vers le dossier des masks')
    parser.add_argument('--images_dir', type=str, default=None,
                        help='Chemin vers le dossier des images (optionnel, pour vérifier correspondance)')
    
    args = parser.parse_args()
    
    check_masks(args.masks_dir, args.images_dir)


if __name__ == "__main__":
    main()
