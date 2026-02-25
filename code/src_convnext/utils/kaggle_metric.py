"""
Kaggle Scientific Image Forgery Detection - Official Metric Implementation

This module implements the exact oF1 (Optimal F1) metric used in the competition.

Key differences from standard pixel-level F1:
1. Instance-level matching: Each predicted region is matched to a GT region
2. Hungarian algorithm: Finds optimal assignment between predictions and GT
3. Penalty for excess predictions
4. Special handling of "authentic" (non-forged) images

Reference: Kaggle competition metric.py
"""

import json
from typing import List, Tuple, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


# =============================================================================
# RLE Encoding/Decoding (matching Kaggle format exactly)
# =============================================================================

def rle_encode_single(mask: npt.NDArray, fg_val: int = 1) -> List[int]:
    """
    RLE encode a single binary mask.
    
    Args:
        mask: Binary mask of shape (height, width)
        fg_val: Foreground value (default 1)
        
    Returns:
        List of [start, length, start, length, ...] pairs
    """
    # Flatten in column-major order (Fortran style) as per Kaggle
    dots = np.where(mask.T.flatten() == fg_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((int(b + 1), 0))  # Convert to native int
        if run_lengths:
            run_lengths[-1] += 1
        prev = b
    # Ensure all values are native Python ints for JSON serialization
    return [int(x) for x in run_lengths]


def rle_encode(masks: List[npt.NDArray], fg_val: int = 1) -> str:
    """
    Encode multiple masks as RLE string (Kaggle format).
    
    Args:
        masks: List of binary masks, each of shape (height, width)
        fg_val: Foreground value
        
    Returns:
        RLE string with each mask JSON-encoded, separated by semicolons
    """
    if not masks:
        return "authentic"
    
    encoded = []
    for mask in masks:
        rle = rle_encode_single(mask, fg_val)
        if rle:  # Only add non-empty masks
            encoded.append(json.dumps(rle))
    
    if not encoded:
        return "authentic"
    
    return ';'.join(encoded)


def rle_decode_single(mask_rle: str, shape: Tuple[int, int]) -> npt.NDArray:
    """
    Decode a single RLE string to binary mask.
    
    Args:
        mask_rle: RLE as JSON string "[start, length, start, length, ...]"
        shape: (height, width) of output mask
        
    Returns:
        Binary mask of shape (height, width)
    """
    if mask_rle == "authentic" or not mask_rle:
        return np.zeros(shape, dtype=np.uint8)
    
    rle_data = json.loads(mask_rle)
    rle_array = np.asarray(rle_data, dtype=np.int32)
    
    if len(rle_array) == 0:
        return np.zeros(shape, dtype=np.uint8)
    
    starts = rle_array[0::2] - 1  # Convert to 0-indexed
    lengths = rle_array[1::2]
    ends = starts + lengths
    
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(shape, order='F')  # Column-major (Fortran) order


def rle_decode(rle_string: str, shape: Tuple[int, int]) -> List[npt.NDArray]:
    """
    Decode RLE string containing multiple masks.
    
    Args:
        rle_string: Semicolon-separated RLE strings or "authentic"
        shape: (height, width) of output masks
        
    Returns:
        List of binary masks
    """
    if rle_string == "authentic" or not rle_string:
        return []
    
    masks = []
    for rle in rle_string.split(';'):
        mask = rle_decode_single(rle.strip(), shape)
        if mask.sum() > 0:  # Only add non-empty masks
            masks.append(mask)
    
    return masks


# =============================================================================
# Instance Detection from Segmentation Mask
# =============================================================================

def mask_to_instances(mask: npt.NDArray, min_area: int = 10) -> List[npt.NDArray]:
    """
    Convert a binary segmentation mask to a list of instance masks.
    Uses connected component analysis.
    
    Args:
        mask: Binary mask of shape (height, width)
        min_area: Minimum area (in pixels) for a valid instance
        
    Returns:
        List of binary masks, one per detected instance
    """
    if mask.sum() == 0:
        return []
    
    # Ensure binary
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Find connected components
    labeled_array, num_features = ndimage.label(binary_mask)
    
    instances = []
    for i in range(1, num_features + 1):
        instance_mask = (labeled_array == i).astype(np.uint8)
        if instance_mask.sum() >= min_area:
            instances.append(instance_mask)
    
    return instances


def merge_nearby_instances(instances: List[npt.NDArray], 
                           distance_threshold: int = 20) -> List[npt.NDArray]:
    """
    Merge instances that are close to each other.
    Useful when copy-move detection fragments a single region.
    
    Args:
        instances: List of binary masks
        distance_threshold: Maximum distance between instances to merge
        
    Returns:
        List of merged binary masks
    """
    if len(instances) <= 1:
        return instances
    
    # Dilate each instance and check for overlap
    from scipy.ndimage import binary_dilation
    
    merged = []
    used = set()
    
    for i, inst_i in enumerate(instances):
        if i in used:
            continue
        
        # Dilate instance
        dilated = binary_dilation(inst_i, iterations=distance_threshold)
        
        # Find overlapping instances
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
# F1 Score Calculation (Pixel-level, for matrix computation)
# =============================================================================

def calculate_f1_score(pred_mask: npt.NDArray, gt_mask: npt.NDArray) -> float:
    """
    Calculate pixel-level F1 score between two binary masks.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        F1 score (0 to 1)
    """
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
    """
    Calculate F1 score matrix between all pairs of predicted and GT masks.
    
    Args:
        pred_masks: List of predicted binary masks
        gt_masks: List of ground truth binary masks
        
    Returns:
        F1 matrix of shape (num_pred, num_gt)
    """
    num_pred = len(pred_masks)
    num_gt = len(gt_masks)
    
    if num_pred == 0 or num_gt == 0:
        return np.zeros((max(num_pred, 1), max(num_gt, 1)))
    
    f1_matrix = np.zeros((num_pred, num_gt))
    
    for i in range(num_pred):
        for j in range(num_gt):
            f1_matrix[i, j] = calculate_f1_score(pred_masks[i], gt_masks[j])
    
    # Pad with zeros if pred < gt (to handle unmatched GT)
    if num_pred < num_gt:
        padding = np.zeros((num_gt - num_pred, num_gt))
        f1_matrix = np.vstack((f1_matrix, padding))
    
    return f1_matrix


# =============================================================================
# Optimal F1 Score (oF1) - Main Metric
# =============================================================================

def oF1_score(pred_masks: List[npt.NDArray], 
              gt_masks: List[npt.NDArray]) -> float:
    """
    Calculate the Optimal F1 score using Hungarian algorithm.
    
    This is the main metric used in the Kaggle competition.
    It finds the optimal assignment of predicted masks to ground truth masks
    and penalizes excess predictions.
    
    Args:
        pred_masks: List of predicted binary masks (instances)
        gt_masks: List of ground truth binary masks (instances)
        
    Returns:
        Optimal F1 score (0 to 1)
    """
    # Handle edge cases
    if len(gt_masks) == 0 and len(pred_masks) == 0:
        return 1.0  # Both authentic
    
    if len(gt_masks) == 0:
        return 0.0  # GT is authentic but we predicted forgery
    
    if len(pred_masks) == 0:
        return 0.0  # GT has forgery but we predicted authentic
    
    # Calculate F1 matrix
    f1_matrix = calculate_f1_matrix(pred_masks, gt_masks)
    
    # Hungarian algorithm for optimal assignment
    # We negate because linear_sum_assignment minimizes
    row_ind, col_ind = linear_sum_assignment(-f1_matrix)
    
    # Calculate mean F1 of matched pairs
    matched_f1 = f1_matrix[row_ind, col_ind]
    mean_f1 = np.mean(matched_f1)
    
    # Penalty for excess predictions
    # This penalizes predicting more instances than GT has
    excess_penalty = len(gt_masks) / max(len(pred_masks), len(gt_masks))
    
    return mean_f1 * excess_penalty


def evaluate_single_image(pred_rle: str, 
                          gt_rle: str, 
                          shape: Tuple[int, int]) -> float:
    """
    Evaluate a single image prediction against ground truth.
    
    Args:
        pred_rle: Predicted RLE string (or "authentic")
        gt_rle: Ground truth RLE string (or "authentic")
        shape: (height, width) of the image
        
    Returns:
        oF1 score for this image
    """
    # Handle authentic cases
    pred_is_authentic = pred_rle == "authentic" or not pred_rle
    gt_is_authentic = gt_rle == "authentic" or not gt_rle
    
    if pred_is_authentic and gt_is_authentic:
        return 1.0  # Both correctly identified as authentic
    
    if pred_is_authentic != gt_is_authentic:
        return 0.0  # One is authentic, other is not
    
    # Both have forgeries - decode and calculate oF1
    pred_masks = rle_decode(pred_rle, shape)
    gt_masks = rle_decode(gt_rle, shape)
    
    return oF1_score(pred_masks, gt_masks)


# =============================================================================
# High-level API for Model Predictions
# =============================================================================

def convert_segmentation_to_submission(
    segmentation_mask: npt.NDArray,
    threshold: float = 0.5,
    min_area: int = 100,
    merge_distance: int = 0
) -> str:
    """
    Convert a model's segmentation output to Kaggle submission format.
    
    Args:
        segmentation_mask: Model output (probability map or binary mask)
        threshold: Binarization threshold
        min_area: Minimum instance area in pixels
        merge_distance: Distance threshold for merging nearby instances (0 = no merge)
        
    Returns:
        RLE-encoded string for submission
    """
    # Binarize
    if segmentation_mask.max() <= 1.0:
        binary_mask = (segmentation_mask > threshold).astype(np.uint8)
    else:
        binary_mask = (segmentation_mask > threshold * 255).astype(np.uint8)
    
    # Check if authentic (no forgery detected)
    if binary_mask.sum() < min_area:
        return "authentic"
    
    # Extract instances
    instances = mask_to_instances(binary_mask, min_area=min_area)
    
    if not instances:
        return "authentic"
    
    # Optionally merge nearby instances
    if merge_distance > 0:
        instances = merge_nearby_instances(instances, merge_distance)
    
    # Encode to RLE
    return rle_encode(instances)


def calculate_competition_score(
    predictions: List[str],
    ground_truths: List[str],
    shapes: List[Tuple[int, int]]
) -> float:
    """
    Calculate the overall competition score.
    
    Args:
        predictions: List of RLE strings (one per image)
        ground_truths: List of GT RLE strings (one per image)
        shapes: List of (height, width) tuples for each image
        
    Returns:
        Mean oF1 score across all images
    """
    assert len(predictions) == len(ground_truths) == len(shapes)
    
    scores = []
    for pred, gt, shape in zip(predictions, ground_truths, shapes):
        score = evaluate_single_image(pred, gt, shape)
        scores.append(score)
    
    return np.mean(scores)


# =============================================================================
# Validation Metric Calculator (for training loop)
# =============================================================================

class KaggleMetricCalculator:
    """
    Calculator for Kaggle oF1 metric during validation.
    
    Accumulates predictions and calculates the competition metric.
    Matches the official Kaggle metric implementation exactly.
    """
    
    def __init__(self, 
                 threshold: float = 0.5, 
                 min_area: int = 100,
                 merge_distance: int = 0):
        self.threshold = threshold
        self.min_area = min_area
        self.merge_distance = merge_distance
        self.reset()
    
    def reset(self):
        """Reset accumulated data."""
        self.predictions = []
        self.ground_truths = []
        self.shapes = []
        self.image_scores = []
        # Debug counters
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
        """
        Update with a single image prediction.
        
        Args:
            pred_mask: Model output (logits or probabilities), shape (H, W) or (1, H, W)
            gt_mask: Ground truth mask, shape (H, W) or (1, H, W)
            original_shape: Original image shape (if different from mask shape)
        """
        # Handle tensor input
        if hasattr(pred_mask, 'cpu'):
            pred_mask = pred_mask.cpu().numpy()
        if hasattr(gt_mask, 'cpu'):
            gt_mask = gt_mask.cpu().numpy()
        
        # Remove channel dimension if present
        if pred_mask.ndim == 3:
            pred_mask = pred_mask.squeeze(0)
        if gt_mask.ndim == 3:
            gt_mask = gt_mask.squeeze(0)
        
        # Apply sigmoid if needed (convert logits to probabilities)
        if pred_mask.min() < 0:
            pred_mask = 1 / (1 + np.exp(-pred_mask))
        
        shape = original_shape or pred_mask.shape
        
        # Convert prediction to submission format
        pred_rle = convert_segmentation_to_submission(
            pred_mask, 
            threshold=self.threshold,
            min_area=self.min_area,
            merge_distance=self.merge_distance
        )
        
        # Convert GT to submission format
        # Use min_area=1 to keep all GT instances (even small ones)
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
        """
        Update with a batch of predictions.
        
        Args:
            pred_batch: Batch of model outputs, shape (B, 1, H, W) or (B, H, W)
            gt_batch: Batch of ground truths, shape (B, 1, H, W) or (B, H, W)
            original_shapes: List of original shapes per image
        """
        if hasattr(pred_batch, 'cpu'):
            pred_batch = pred_batch.cpu().numpy()
        if hasattr(gt_batch, 'cpu'):
            gt_batch = gt_batch.cpu().numpy()
        
        batch_size = pred_batch.shape[0]
        
        for i in range(batch_size):
            orig_shape = original_shapes[i] if original_shapes else None
            self.update(pred_batch[i], gt_batch[i], orig_shape)
    
    def compute(self) -> dict:
        """
        Compute final metrics.
        
        Returns:
            Dictionary with oF1 score and additional statistics
        """
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
# Testing
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