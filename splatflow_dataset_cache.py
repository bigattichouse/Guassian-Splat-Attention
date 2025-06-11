#!/usr/bin/env python3
"""
SplatFlow Dataset Pre-loader and Cache Manager
Forces caching of all SplatFlow datasets ahead of time.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
    from transformers import GPT2Tokenizer
    DATASETS_AVAILABLE = True
    print("✅ Datasets and transformers available")
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install datasets transformers")
    sys.exit(1)


class SplatFlowDatasetPreloader:
    """Pre-loads and caches all SplatFlow datasets"""
    
    def __init__(self):
        self.cache_dir = Path.home() / '.cache' / 'huggingface' / 'datasets'
        self.tokenizer = None
        
        # SplatFlow dataset configurations
        self.splatflow_datasets = {
            "roneneldan/TinyStories": {
                "name": "TinyStories",
                "description": "High-quality children's stories",
                "split": "train",
                "text_field": "text", 
                "target_count": 3000,
                "streaming": True,
                "config": None
            },
            "squad": {
                "name": "SQuAD",
                "description": "High-quality Wikipedia paragraphs", 
                "split": "train",
                "text_field": "context",
                "target_count": 2000,
                "streaming": False,
                "config": None
            },
            "ag_news": {
                "name": "AG News",
                "description": "News articles",
                "split": "train", 
                "text_field": "text",
                "target_count": 1500,
                "streaming": False,
                "config": None
            },
            "cnn_dailymail": {
                "name": "CNN/DailyMail",
                "description": "Professional news articles",
                "split": "train",
                "text_field": "article", 
                "target_count": 2000,
                "streaming": True,
                "config": "3.0.0"
            },
            "openwebtext": {
                "name": "OpenWebText", 
                "description": "Curated web content",
                "split": "train",
                "text_field": "text",
                "target_count": 2500, 
                "streaming": True,
                "config": None
            },
            "bookcorpus": {
                "name": "BookCorpus",
                "description": "Literature excerpts", 
                "split": "train",
                "text_field": "text",
                "target_count": 2000,
                "streaming": True,
                "config": None
            },
            "multi_news": {
                "name": "MultiNews",
                "description": "Multi-document news",
                "split": "train", 
                "text_field": "document",
                "target_count": 800,
                "streaming": False,
                "config": None
            }
        }
        
    def setup_tokenizer(self):
        """Setup tokenizer for validation"""
        try:
            logger.info("Setting up tokenizer...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("✅ Tokenizer ready")
        except Exception as e:
            logger.warning(f"Failed to setup tokenizer: {e}")
            self.tokenizer = None
    
    def check_cache_status(self) -> Dict:
        """Check which datasets are already cached"""
        logger.info("🔍 Checking cache status...")
        
        cached = {}
        
        for dataset_id, config in self.splatflow_datasets.items():
            name = config["name"]
            
            # Check various possible cache directory names
            possible_cache_names = [
                dataset_id.replace("/", "___"),
                dataset_id.replace("/", "--"),
                dataset_id.replace("/", "_"),
                dataset_id.lower().replace("/", "___"),
                dataset_id.lower().replace("/", "--")
            ]
            
            is_cached = False
            cache_path = None
            
            for cache_name in possible_cache_names:
                potential_path = self.cache_dir / cache_name
                if potential_path.exists():
                    is_cached = True
                    cache_path = potential_path
                    break
            
            cached[dataset_id] = {
                "name": name,
                "is_cached": is_cached,
                "cache_path": cache_path
            }
            
            status = "✅ CACHED" if is_cached else "❌ NOT CACHED"
            logger.info(f"   {name}: {status}")
        
        cached_count = sum(1 for info in cached.values() if info["is_cached"])
        total_count = len(cached)
        
        logger.info(f"📊 Cache summary: {cached_count}/{total_count} datasets cached")
        
        return cached
    
    def estimate_download_size(self) -> str:
        """Estimate total download size"""
        # Rough estimates based on typical dataset sizes
        size_estimates = {
            "roneneldan/TinyStories": 500,  # MB
            "squad": 50,
            "ag_news": 30, 
            "cnn_dailymail": 800,
            "openwebtext": 1200,
            "bookcorpus": 600,
            "multi_news": 200
        }
        
        total_mb = sum(size_estimates.values())
        total_gb = total_mb / 1024
        
        return f"~{total_gb:.1f}GB ({total_mb}MB)"
    
    def preload_dataset(self, dataset_id: str, config: Dict, force_reload: bool = False) -> bool:
        """Pre-load a single dataset"""
        name = config["name"]
        
        logger.info(f"📥 Pre-loading {name}...")
        logger.info(f"    Dataset: {dataset_id}")
        logger.info(f"    Description: {config['description']}")
        
        try:
            start_time = time.time()
            
            # Load dataset with same parameters as SplatFlow training
            load_kwargs = {
                "path": dataset_id,
                "split": config["split"],
                "streaming": config["streaming"]
            }
            
            if config["config"]:
                load_kwargs["name"] = config["config"]
            
            dataset = load_dataset(**load_kwargs)
            
            # Sample some data to ensure it's actually cached
            text_field = config["text_field"]
            target_count = min(config["target_count"], 100)  # Just sample for caching
            
            sampled_count = 0
            valid_texts = 0
            
            for item in dataset:
                if sampled_count >= target_count:
                    break
                
                try:
                    text = item.get(text_field, "")
                    
                    if isinstance(text, list) and len(text) > 0:
                        text = text[0] if len(text) == 1 else " | ".join([t for t in text if t and len(t.strip()) > 20])
                    
                    if isinstance(text, str) and len(text.strip()) > 50:
                        valid_texts += 1
                        
                        # Quick tokenization test if tokenizer available
                        if self.tokenizer and sampled_count < 5:
                            try:
                                tokens = self.tokenizer.encode(text[:500], max_length=100, truncation=True)
                                if len(tokens) < 5:
                                    logger.warning(f"Very short tokenization for {name}")
                            except Exception:
                                pass
                    
                    sampled_count += 1
                    
                    # Progress update
                    if sampled_count % 20 == 0:
                        logger.info(f"      Processed {sampled_count} samples...")
                        
                except Exception as e:
                    logger.warning(f"Error processing item in {name}: {e}")
                    continue
            
            load_time = time.time() - start_time
            
            logger.info(f"    ✅ {name} loaded successfully!")
            logger.info(f"       Processed: {sampled_count} samples")
            logger.info(f"       Valid texts: {valid_texts}")
            logger.info(f"       Time: {load_time:.1f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"    ❌ Failed to load {name}: {e}")
            return False
    
    def preload_all_datasets(self, force_reload: bool = False, skip_cached: bool = True) -> Dict:
        """Pre-load all SplatFlow datasets"""
        
        logger.info("🚀 Starting SplatFlow dataset pre-loading...")
        logger.info(f"   Estimated download size: {self.estimate_download_size()}")
        logger.info(f"   Force reload: {force_reload}")
        logger.info(f"   Skip cached: {skip_cached}")
        
        # Check current cache status
        cache_status = self.check_cache_status()
        
        # Setup tokenizer for validation
        self.setup_tokenizer()
        
        results = {
            "success": [],
            "failed": [],
            "skipped": []
        }
        
        total_datasets = len(self.splatflow_datasets)
        current_idx = 0
        
        for dataset_id, config in self.splatflow_datasets.items():
            current_idx += 1
            name = config["name"]
            
            logger.info(f"\n📦 [{current_idx}/{total_datasets}] Processing {name}")
            
            # Check if should skip
            if skip_cached and cache_status[dataset_id]["is_cached"] and not force_reload:
                logger.info(f"    ⏭️  Skipping {name} (already cached)")
                results["skipped"].append(name)
                continue
            
            # Pre-load the dataset
            success = self.preload_dataset(dataset_id, config, force_reload)
            
            if success:
                results["success"].append(name)
            else:
                results["failed"].append(name)
            
            # Brief pause between datasets
            time.sleep(1)
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("📊 Pre-loading Summary:")
        logger.info(f"   ✅ Successful: {len(results['success'])} datasets")
        logger.info(f"   ❌ Failed: {len(results['failed'])} datasets") 
        logger.info(f"   ⏭️  Skipped: {len(results['skipped'])} datasets")
        
        if results["success"]:
            logger.info("   Successful datasets:")
            for name in results["success"]:
                logger.info(f"     • {name}")
        
        if results["failed"]:
            logger.warning("   Failed datasets:")
            for name in results["failed"]:
                logger.warning(f"     • {name}")
        
        if results["skipped"]:
            logger.info("   Skipped datasets:")
            for name in results["skipped"]:
                logger.info(f"     • {name}")
        
        return results
    
    def clear_cache(self, dataset_names: List[str] = None):
        """Clear cache for specific datasets or all"""
        
        if dataset_names is None:
            logger.info("🗑️  Clearing ALL dataset cache...")
            if self.cache_dir.exists():
                import shutil
                shutil.rmtree(self.cache_dir)
                logger.info("   ✅ All cache cleared")
            else:
                logger.info("   ℹ️  No cache directory found")
        else:
            logger.info(f"🗑️  Clearing cache for specific datasets: {dataset_names}")
            # Implementation for specific dataset clearing would go here
            # This is more complex as we need to match cache directory names
            logger.warning("   ⚠️  Specific dataset clearing not yet implemented")
            logger.info("   Use: rm -rf ~/.cache/huggingface/datasets/[dataset_name]")


def main():
    """Main pre-loader interface"""
    parser = argparse.ArgumentParser(description="SplatFlow Dataset Pre-loader and Cache Manager")
    
    parser.add_argument("--action", "-a",
                       choices=["check", "preload", "clear", "force-preload"],
                       default="check",
                       help="Action to perform")
    
    parser.add_argument("--skip-cached", action="store_true", default=True,
                       help="Skip datasets that are already cached (default: True)")
    
    parser.add_argument("--force", action="store_true",
                       help="Force re-download even if cached")
    
    args = parser.parse_args()
    
    print("🚀 SplatFlow Dataset Pre-loader and Cache Manager")
    print("=" * 60)
    
    preloader = SplatFlowDatasetPreloader()
    
    if args.action == "check":
        print("🔍 Checking cache status...")
        cache_status = preloader.check_cache_status()
        
        cached_count = sum(1 for info in cache_status.values() if info["is_cached"])
        total_count = len(cache_status)
        
        if cached_count == total_count:
            print("\n🎉 All SplatFlow datasets are cached!")
            print("   Your next training run will be fast.")
        elif cached_count > 0:
            print(f"\n🔄 {cached_count}/{total_count} datasets cached.")
            print("   Run with --action preload to cache remaining datasets.")
        else:
            print("\n📥 No datasets cached.")
            print("   Run with --action preload to cache all datasets.")
            print(f"   Estimated download size: {preloader.estimate_download_size()}")
    
    elif args.action in ["preload", "force-preload"]:
        force_reload = args.force or args.action == "force-preload"
        
        print(f"📥 Pre-loading SplatFlow datasets...")
        print(f"   Force reload: {force_reload}")
        print(f"   Skip cached: {args.skip_cached}")
        print(f"   Estimated download: {preloader.estimate_download_size()}")
        
        results = preloader.preload_all_datasets(
            force_reload=force_reload,
            skip_cached=args.skip_cached
        )
        
        if len(results["failed"]) == 0:
            print("\n🎉 All datasets successfully cached!")
            print("   Your SplatFlow training will now be much faster.")
        else:
            print(f"\n⚠️  {len(results['failed'])} datasets failed to cache.")
            print("   Training will still work but may download during training.")
    
    elif args.action == "clear":
        print("🗑️  Clearing dataset cache...")
        print("⚠️  This will delete all cached datasets!")
        
        confirm = input("Are you sure? (y/N): ")
        if confirm.lower() in ['y', 'yes']:
            preloader.clear_cache()
            print("✅ Cache cleared. Next run will download all datasets.")
        else:
            print("❌ Cache clear cancelled.")
    
    print("\n" + "=" * 60)
    print("💡 Usage Tips:")
    print("   • Run 'python script.py --action preload' to cache all datasets")
    print("   • Run 'python script.py --action check' to see cache status")
    print("   • Cached datasets make training much faster")
    print("   • Safe to delete cache if you need disk space")


if __name__ == "__main__":
    main()
