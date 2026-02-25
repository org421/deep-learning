"""Metrique oF1 du concours Kaggle - calcul au niveau instance avec algorithme hongrois."""

import json
from typing import List, Tuple, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


# =============================================================================
# Encodage/decodage RLE (format Kaggle)
# =============================================================================

def rle_encode_single(mask: npt.NDArray, fg_val: int = 1) -> List[int]:
    """Encode un masque binaire en RLE."""
    # ordre colonne-major (Fortran) comme Kaggle
    dots = np.where(mask.T.flatten() == fg_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((int(b + 1), 0))  # conversion en int natif
        if run_lengths:
            run_lengths[-1] += 1
        prev = b
    # int python natifs pour le JSON
    return [int(x) for x in run_lengths]


def rle_encode(masks: List[npt.NDArray], fg_val: int = 1) -> str:
    """Encode plusieurs masques en RLE (format Kaggle)."""
    if not masks:
        return "authentic"
    
    encoded = []
    for mask in masks:
        rle = rle_encode_single(mask, fg_val)
        if rle:  # que les masques non vides
            encoded.append(json.dumps(rle))
    
    if not encoded:
        return "authentic"
    
    return ';'.join(encoded)


def rle_decode_single(mask_rle: str, shape: Tuple[int, int]) -> npt.NDArray:
    """Decode un RLE en masque binaire."""
    if mask_rle == "authentic" or not mask_rle:
        return np.zeros(shape, dtype=np.uint8)
    
    rle_data = json.loads(mask_rle)
    rle_array = np.asarray(rle_data, dtype=np.int32)
    
    if len(rle_array) == 0:
        return np.zeros(shape, dtype=np.uint8)
    
    starts = rle_array[0::2] - 1  # passage en 0-indexed
    lengths = rle_array[1::2]
    ends = starts + lengths
    
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(shape, order='F')  # ordre Fortran


def rle_decode(rle_string: str, shape: Tuple[int, int]) -> List[npt.NDArray]:
    """Decode un string RLE contenant plusieurs masques."""
    if rle_string == "authentic" or not rle_string:
        return []
    
    masks = []
    for rle in rle_string.split(';'):
        mask = rle_decode_single(rle.strip(), shape)
        if mask.sum() > 0:  # que les masques non vides
            masks.append(mask)
    
    return masks


# =============================================================================
# Detection d'instances
# =============================================================================

def mask_to_instances(mask: npt.NDArray, min_area: int = 10) -> List[npt.NDArray]:
    """Separe un masque en instances via composantes connexes."""
    if mask.sum() == 0:
        return []
    
    # binarisation
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # composantes connexes
    labeled_array, num_features = ndimage.label(binary_mask)
    
    instances = []
    for i in range(1, num_features + 1):
        instance_mask = (labeled_array == i).astype(np.uint8)
        if instance_mask.sum() >= min_area:
            instances.append(instance_mask)
    
    return instances


def merge_nearby_instances(instances: List[npt.NDArray], 
                           distance_threshold: int = 20) -> List[npt.NDArray]:
    """Fusionne les instances proches."""
    if len(instances) <= 1:
        return instances
    
    # dilatation pour detecter les chevauchements
    from scipy.ndimage import binary_dilation
    
    merged = []
    used = set()
    
    for i, inst_i in enumerate(instances):
        if i in used:
            continue
        
        # dilatation
        dilated = binary_dilation(inst_i, iterations=distance_threshold)
        
        # instances qui se chevauchent
        current_merge = inst_i.copy()
        for j, inst_j in enumerate(instances):
            if j <= i or j in used:
                continue
            
            if np.any(dilated & inst_j):
                current_merge = np.logical_or(current_merge, inst_j).astype(np.uint8)
                used.add(j)
        
        merged.append(current_merge)
        used.add(i)
    
    return merged


# =============================================================================
# Calcul du F1 pixel
# =============================================================================

def calculate_f1_score(pred_mask: npt.NDArray, gt_mask: npt.NDArray) -> float:
    """F1 pixel entre deux masques."""
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if (precision + recall) > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0


def calculate_f1_matrix(pred_masks: List[npt.NDArray], 
                        gt_masks: List[npt.NDArray]) -> npt.NDArray:
    """Matrice de F1 entre predictions et GT."""
    num_pred = len(pred_masks)
    num_gt = len(gt_masks)
    
    if num_pred == 0 or num_gt == 0:
        return np.zeros((max(num_pred, 1), max(num_gt, 1)))
    
    f1_matrix = np.zeros((num_pred, num_gt))
    
    for i in range(num_pred):
        for j in range(num_gt):
            f1_matrix[i, j] = calculate_f1_score(pred_masks[i], gt_masks[j])
    
    # padding si moins de preds que de GT
    if num_pred < num_gt:
        padding = np.zeros((num_gt - num_pred, num_gt))
        f1_matrix = np.vstack((f1_matrix, padding))
    
    return f1_matrix


# =============================================================================
# Score oF1 - metrique principale
# =============================================================================

def oF1_score(pred_masks: List[npt.NDArray], 
              gt_masks: List[npt.NDArray]) -> float:
    """Score oF1 avec matching hongrois."""
    # cas limites
    if len(gt_masks) == 0 and len(pred_masks) == 0:
        return 1.0  # les deux authentiques
    
    if len(gt_masks) == 0:
        return 0.0  # GT authentique mais on a predit forgery
    
    if len(pred_masks) == 0:
        return 0.0  # GT forge mais predit authentique
    
    # matrice F1
    f1_matrix = calculate_f1_matrix(pred_masks, gt_masks)
    
    # algo hongrois
    # on inverse car linear_sum_assignment minimise
    row_ind, col_ind = linear_sum_assignment(-f1_matrix)
    
    # F1 moyen des paires matchees
    matched_f1 = f1_matrix[row_ind, col_ind]
    mean_f1 = np.mean(matched_f1)
    
    # penalite pour les predictions en trop
    excess_penalty = len(gt_masks) / max(len(pred_masks), len(gt_masks))
    
    return mean_f1 * excess_penalty


def evaluate_single_image(pred_rle: str, 
                          gt_rle: str, 
                          shape: Tuple[int, int]) -> float:
    """Evalue une image (pred vs GT)."""
    # cas authentique
    pred_is_authentic = pred_rle == "authentic" or not pred_rle
    gt_is_authentic = gt_rle == "authentic" or not gt_rle
    
    if pred_is_authentic and gt_is_authentic:
        return 1.0  # Both correctly identified as authentic
    
    if pred_is_authentic != gt_is_authentic:
        return 0.0  # One is authentic, other is not
    
    # les deux ont des forgeries, on calcule le oF1
    pred_masks = rle_decode(pred_rle, shape)
    gt_masks = rle_decode(gt_rle, shape)
    
    return oF1_score(pred_masks, gt_masks)


# =============================================================================
# API pour les predictions
# =============================================================================

def convert_segmentation_to_submission(
    segmentation_mask: npt.NDArray,
    threshold: float = 0.5,
    min_area: int = 100,
    merge_distance: int = 0
) -> str:
    """Convertit un masque de segmentation au format soumission Kaggle."""
    # binarisation
    if segmentation_mask.max() <= 1.0:
        binary_mask = (segmentation_mask > threshold).astype(np.uint8)
    else:
        binary_mask = (segmentation_mask > threshold * 255).astype(np.uint8)
    
    # authentique si rien detecte
    if binary_mask.sum() < min_area:
        return "authentic"
    
    # extraction des instances
    instances = mask_to_instances(binary_mask, min_area=min_area)
    
    if not instances:
        return "authentic"
    
    # fusion des instances proches si active
    if merge_distance > 0:
        instances = merge_nearby_instances(instances, merge_distance)
    
    # encodage RLE
    return rle_encode(instances)


def calculate_competition_score(
    predictions: List[str],
    ground_truths: List[str],
    shapes: List[Tuple[int, int]]
) -> float:
    """Score moyen oF1 sur toutes les images."""
    assert len(predictions) == len(ground_truths) == len(shapes)
    
    scores = []
    for pred, gt, shape in zip(predictions, ground_truths, shapes):
        score = evaluate_single_image(pred, gt, shape)
        scores.append(score)
    
    return np.mean(scores)


# =============================================================================
# Calculateur pour la validation
# =============================================================================

class KaggleMetricCalculator:
    """Calcul du oF1 pendant la validation."""
    
    def __init__(self, 
                 threshold: float = 0.5, 
                 min_area: int = 100,
                 merge_distance: int = 0):
        self.threshold = threshold
        self.min_area = min_area
        self.merge_distance = merge_distance
        self.reset()
    
    def reset(self):
        """Reinitialise."""
        self.predictions = []
        self.ground_truths = []
        self.shapes = []
        self.image_scores = []
        # compteurs debug
        self._debug_gt_authentic = 0
        self._debug_gt_forged = 0
        self._debug_pred_authentic = 0
        self._debug_pred_forged = 0
        self._debug_both_authentic = 0
        self._debug_both_forged = 0
        self._debug_gt_forged_pred_authentic = 0
        self._debug_gt_authentic_pred_forged = 0
    
    def update(self, 
               pred_mask: npt.NDArray, 
               gt_mask: npt.NDArray,
               original_shape: Optional[Tuple[int, int]] = None):
        """Met a jour avec une prediction."""
        # gestion des tensors
        if hasattr(pred_mask, 'cpu'):
            pred_mask = pred_mask.cpu().numpy()
        if hasattr(gt_mask, 'cpu'):
            gt_mask = gt_mask.cpu().numpy()
        
        # enleve la dim channel
        if pred_mask.ndim == 3:
            pred_mask = pred_mask.squeeze(0)
        if gt_mask.ndim == 3:
            gt_mask = gt_mask.squeeze(0)
        
        # sigmoid si necessaire
        if pred_mask.min() < 0:
            pred_mask = 1 / (1 + np.exp(-pred_mask))
        
        shape = original_shape or pred_mask.shape
        
        # conversion au format soumission
        pred_rle = convert_segmentation_to_submission(
            pred_mask, 
            threshold=self.threshold,
            min_area=self.min_area,
            merge_distance=self.merge_distance
        )
        
        # GT au format soumission (on garde tout)
        gt_rle = convert_segmentation_to_submission(
            gt_mask.astype(np.float32),
            threshold=0.5,
            min_area=1,
            merge_distance=0
        )
        
        # Calculate score using official Kaggle logic
        pred_is_authentic = (pred_rle == "authentic")
        gt_is_authentic = (gt_rle == "authentic")
        
        # Update debug counters
        if gt_is_authentic:
            self._debug_gt_authentic += 1
        else:
            self._debug_gt_forged += 1
        
        if pred_is_authentic:
            self._debug_pred_authentic += 1
        else:
            self._debug_pred_forged += 1
        
        # Official Kaggle scoring logic:
        # If either is authentic, score is 1.0 only if both match, else 0.0
        if pred_is_authentic or gt_is_authentic:
            if pred_is_authentic and gt_is_authentic:
                score = 1.0
                self._debug_both_authentic += 1
            else:
                score = 0.0
                if gt_is_authentic and not pred_is_authentic:
                    self._debug_gt_authentic_pred_forged += 1
                else:
                    self._debug_gt_forged_pred_authentic += 1
        else:
            # Both have forgeries - calculate oF1
            self._debug_both_forged += 1
            score = evaluate_single_image(pred_rle, gt_rle, shape)
        
        self.predictions.append(pred_rle)
        self.ground_truths.append(gt_rle)
        self.shapes.append(shape)
        self.image_scores.append(score)
    
    def update_batch(self, 
                     pred_batch: npt.NDArray, 
                     gt_batch: npt.NDArray,
                     original_shapes: Optional[List[Tuple[int, int]]] = None):
        """Met a jour avec un batch."""
        if hasattr(pred_batch, 'cpu'):
            pred_batch = pred_batch.cpu().numpy()
        if hasattr(gt_batch, 'cpu'):
            gt_batch = gt_batch.cpu().numpy()
        
        batch_size = pred_batch.shape[0]
        
        for i in range(batch_size):
            orig_shape = original_shapes[i] if original_shapes else None
            self.update(pred_batch[i], gt_batch[i], orig_shape)
    
    def compute(self) -> dict:
        """Calcule les metriques finales."""
        if not self.image_scores:
            return {'oF1': 0.0, 'num_images': 0}
        
        oF1 = np.mean(self.image_scores)
        
        # Count authentic predictions/labels
        num_pred_authentic = sum(1 for p in self.predictions if p == "authentic")
        num_gt_authentic = sum(1 for g in self.ground_truths if g == "authentic")
        
        # Count correct authentic predictions
        correct_authentic = sum(
            1 for p, g in zip(self.predictions, self.ground_truths)
            if p == "authentic" and g == "authentic"
        )
        
        # Calculate forged-only oF1 (excluding authentic images)
        forged_scores = [
            score for score, gt in zip(self.image_scores, self.ground_truths)
            if gt != "authentic"
        ]
        forged_oF1 = np.mean(forged_scores) if forged_scores else 0.0
        
        return {
            'oF1': oF1,
            'forged_oF1': forged_oF1,  # oF1 on forged images only (more meaningful!)
            'num_images': len(self.image_scores),
            'num_pred_authentic': num_pred_authentic,
            'num_gt_authentic': num_gt_authentic,
            'correct_authentic': correct_authentic,
            'min_score': np.min(self.image_scores),
            'max_score': np.max(self.image_scores),
            'std_score': np.std(self.image_scores),
            # Debug stats
            'debug_gt_authentic': self._debug_gt_authentic,
            'debug_gt_forged': self._debug_gt_forged,
            'debug_pred_authentic': self._debug_pred_authentic,
            'debug_pred_forged': self._debug_pred_forged,
            'debug_both_authentic': self._debug_both_authentic,
            'debug_both_forged': self._debug_both_forged,
            'debug_gt_forged_pred_authentic': self._debug_gt_forged_pred_authentic,
            'debug_gt_authentic_pred_forged': self._debug_gt_authentic_pred_forged,
        }


# =============================================================================
# Tests
# =============================================================================

if __name__ == '__main__':
    print("Testing Kaggle oF1 metric implementation...")
    
    # Test 1: Both authentic
    print("\n1. Both authentic:")
    score = evaluate_single_image("authentic", "authentic", (100, 100))
    print(f"   Score: {score} (expected: 1.0)")
    
    # Test 2: GT authentic, pred has forgery
    print("\n2. GT authentic, pred has forgery:")
    pred_mask = np.zeros((100, 100), dtype=np.uint8)
    pred_mask[20:40, 20:40] = 1
    pred_rle = rle_encode([pred_mask])
    score = evaluate_single_image(pred_rle, "authentic", (100, 100))
    print(f"   Score: {score} (expected: 0.0)")
    
    # Test 3: Perfect match
    print("\n3. Perfect match:")
    gt_mask = np.zeros((100, 100), dtype=np.uint8)
    gt_mask[20:40, 20:40] = 1
    gt_rle = rle_encode([gt_mask])
    score = evaluate_single_image(pred_rle, gt_rle, (100, 100))
    print(f"   Score: {score} (expected: 1.0)")
    
    # Test 4: Partial overlap
    print("\n4. Partial overlap:")
    pred_mask2 = np.zeros((100, 100), dtype=np.uint8)
    pred_mask2[25:45, 25:45] = 1  # Shifted
    pred_rle2 = rle_encode([pred_mask2])
    score = evaluate_single_image(pred_rle2, gt_rle, (100, 100))
    print(f"   Score: {score:.4f} (expected: ~0.4-0.6)")
    
    # Test 5: Multiple instances
    print("\n5. Multiple instances:")
    gt1 = np.zeros((100, 100), dtype=np.uint8)
    gt1[10:20, 10:20] = 1
    gt2 = np.zeros((100, 100), dtype=np.uint8)
    gt2[50:60, 50:60] = 1
    gt_multi_rle = rle_encode([gt1, gt2])
    
    pred1 = gt1.copy()  # Perfect match for first
    pred_single_rle = rle_encode([pred1])  # Only one prediction
    
    score = evaluate_single_image(pred_single_rle, gt_multi_rle, (100, 100))
    print(f"   Score (1 pred, 2 GT): {score:.4f} (expected: ~0.5 due to penalty)")
    
    # Test 6: Excess predictions
    print("\n6. Excess predictions:")
    pred_excess_rle = rle_encode([gt1, gt2, pred_mask2])  # 3 predictions
    score = evaluate_single_image(pred_excess_rle, gt_multi_rle, (100, 100))
    print(f"   Score (3 pred, 2 GT): {score:.4f} (expected: <1.0 due to penalty)")
    
    # Test KaggleMetricCalculator
    print("\n7. Testing KaggleMetricCalculator:")
    calc = KaggleMetricCalculator(threshold=0.5, min_area=10)
    
    # Add some predictions
    for _ in range(5):
        pred = np.random.rand(256, 256)
        gt = (np.random.rand(256, 256) > 0.9).astype(np.float32)
        calc.update(pred, gt)
    
    metrics = calc.compute()
    print(f"   oF1: {metrics['oF1']:.4f}")
    print(f"   Num images: {metrics['num_images']}")
    
    print("\n All tests completed!")