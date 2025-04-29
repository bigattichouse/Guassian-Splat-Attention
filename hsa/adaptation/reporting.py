"""
Adaptation reporting module for Hierarchical Splat Attention (HSA).

This module provides reporting and visualization for adaptation events:
- Detailed reporting on adaptation operations
- Visualization helpers for adaptation events
- History tracking and serialization
- Statistics and analytics on adaptation patterns
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import logging
import time
import json
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import core data structures
from hsa.data_structures import Splat, SplatRegistry
from hsa.adaptation.core import AdaptationType, AdaptationMonitor, AdaptationResult, SplatChange


class AdaptationReporter:
    """
    Reporter for adaptation events, providing detailed analysis and visualization.
    
    This class generates reports on adaptation operations, maintains history,
    and provides visualization helpers for understanding adaptation patterns.
    """
    
    def __init__(
        self, 
        output_dir: str = "hsa_adaptation_reports",
        save_reports: bool = True
    ):
        """
        Initialize the adaptation reporter.
        
        Args:
            output_dir: Directory to save reports
            save_reports: Whether to save reports to disk
        """
        self.output_dir = output_dir
        self.save_reports = save_reports
        self.reports = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if needed
        if self.save_reports:
            os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(
        self, 
        result: AdaptationResult,
        splat_registry: SplatRegistry,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a detailed report from an adaptation result.
        
        Args:
            result: The adaptation result to report on
            splat_registry: The current splat registry
            metrics: Optional additional metrics to include
            
        Returns:
            Dictionary report data
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "cycle_id": len(self.reports),
            "summary": result.get_summary(),
            "changes": self._format_changes(result.changes),
            "registry_state": self._capture_registry_state(splat_registry)
        }
        
        # Add metrics if provided
        if metrics:
            report["metrics"] = metrics
        
        # Save the report
        self.reports.append(report)
        
        if self.save_reports:
            self._save_report(report)
        
        return report
    
    def _format_changes(self, changes: List[SplatChange]) -> List[Dict[str, Any]]:
        """
        Format change records for reporting.
        
        Args:
            changes: List of splat changes
            
        Returns:
            List of formatted change dictionaries
        """
        formatted = []
        
        for change in changes:
            formatted_change = {
                "splat_id": change.splat_id,
                "level": change.level,
                "change_type": change.change_type.value
            }
            
            # Add position data if available
            if change.position_before is not None:
                formatted_change["position_before"] = change.position_before.tolist()
            if change.position_after is not None:
                formatted_change["position_after"] = change.position_after.tolist()
            
            # Add amplitude data if available
            if change.amplitude_before is not None:
                formatted_change["amplitude_before"] = change.amplitude_before
            if change.amplitude_after is not None:
                formatted_change["amplitude_after"] = change.amplitude_after
            
            # Add relationship data
            if change.related_splat_ids:
                formatted_change["related_splat_ids"] = change.related_splat_ids
            if change.created_splat_ids:
                formatted_change["created_splat_ids"] = change.created_splat_ids
            if change.parent_id:
                formatted_change["parent_id"] = change.parent_id
                
            # Add metrics if available
            if change.metrics_before:
                formatted_change["metrics_before"] = change.metrics_before
            
            formatted.append(formatted_change)
        
        return formatted
    
    def _capture_registry_state(self, splat_registry: SplatRegistry) -> Dict[str, Any]:
        """
        Capture the current state of the splat registry.
        
        Args:
            splat_registry: The splat registry
            
        Returns:
            Dictionary of registry state information
        """
        state = {
            "total_splats": len(splat_registry.splats),
            "splats_by_level": {}
        }
        
        # Count splats by level
        for level in splat_registry.hierarchy.levels:
            state["splats_by_level"][level] = len(splat_registry.get_splats_at_level(level))
        
        # Capture splat relationships
        parent_child_counts = {}
        for level in splat_registry.hierarchy.levels:
            parent_child_counts[level] = {
                "avg_children": 0,
                "max_children": 0,
                "total_children": 0,
                "count_with_children": 0
            }
        
        for splat in splat_registry.splats.values():
            children_count = len(splat.children)
            
            if level in parent_child_counts:
                parent_child_counts[level]["total_children"] += children_count
                
                if children_count > 0:
                    parent_child_counts[level]["count_with_children"] += 1
                
                parent_child_counts[level]["max_children"] = max(
                    parent_child_counts[level]["max_children"], 
                    children_count
                )
        
        # Calculate averages
        for level, counts in parent_child_counts.items():
            level_splats = state["splats_by_level"].get(level, 0)
            if level_splats > 0:
                counts["avg_children"] = counts["total_children"] / level_splats
        
        state["parent_child_relationships"] = parent_child_counts
        
        return state
    
    def _save_report(self, report: Dict[str, Any]) -> None:
        """
        Save a report to disk.
        
        Args:
            report: The report data to save
        """
        # Create filename
        filename = f"adaptation_report_{self.session_id}_{report['cycle_id']}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy arrays to lists
        report_serializable = self._make_serializable(report)
        
        # Save as JSON
        try:
            with open(filepath, 'w') as f:
                json.dump(report_serializable, f, indent=2)
            logger.info(f"Saved adaptation report to {filepath}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert an object to a JSON-serializable form.
        
        Args:
            obj: The object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete adaptation history.
        
        Returns:
            List of all reports
        """
        return self.reports
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all adaptation cycles.
        
        Returns:
            Dictionary of summary statistics
        """
        if not self.reports:
            return {"error": "No reports available"}
        
        stats = {
            "total_cycles": len(self.reports),
            "total_changes": sum(len(report["changes"]) for report in self.reports),
            "changes_by_type": {
                "birth": 0,
                "mitosis": 0,
                "death": 0,
                "merge": 0,
                "adjust": 0
            },
            "changes_by_level": {},
            "splat_count_history": []
        }
        
        # Collect all levels across reports
        all_levels = set()
        for report in self.reports:
            for level in report["registry_state"]["splats_by_level"].keys():
                all_levels.add(level)
        
        # Initialize level counts
        for level in all_levels:
            stats["changes_by_level"][level] = {
                "birth": 0,
                "mitosis": 0,
                "death": 0,
                "merge": 0,
                "adjust": 0
            }
        
        # Process each report
        for report in self.reports:
            # Count changes by type
            summary = report["summary"]
            stats["changes_by_type"]["birth"] += summary.get("birth_count", 0)
            stats["changes_by_type"]["mitosis"] += summary.get("mitosis_count", 0)
            stats["changes_by_type"]["death"] += summary.get("death_count", 0)
            stats["changes_by_type"]["merge"] += summary.get("merge_count", 0)
            stats["changes_by_type"]["adjust"] += summary.get("adjust_count", 0)
            
            # Track splat count history
            stats["splat_count_history"].append({
                "cycle": report["cycle_id"],
                "total": report["registry_state"]["total_splats"],
                "by_level": report["registry_state"]["splats_by_level"]
            })
            
            # Count changes by level
            for change in report["changes"]:
                change_type = change["change_type"]
                level = change["level"]
                
                if level in stats["changes_by_level"]:
                    if change_type in stats["changes_by_level"][level]:
                        stats["changes_by_level"][level][change_type] += 1
        
        # Calculate average changes per cycle
        if len(self.reports) > 0:
            stats["avg_changes_per_cycle"] = stats["total_changes"] / len(self.reports)
            for change_type in stats["changes_by_type"]:
                stats[f"avg_{change_type}_per_cycle"] = stats["changes_by_type"][change_type] / len(self.reports)
        
        # Calculate stability metrics
        if len(stats["splat_count_history"]) >= 2:
            # Calculate average change in splat count
            total_diffs = []
            for i in range(1, len(stats["splat_count_history"])):
                prev = stats["splat_count_history"][i-1]["total"]
                curr = stats["splat_count_history"][i]["total"]
                total_diffs.append(abs(curr - prev))
            
            stats["avg_splat_count_change"] = sum(total_diffs) / len(total_diffs) if total_diffs else 0
            stats["stability_score"] = 1.0 - min(1.0, stats["avg_splat_count_change"] / 
                                              (stats["splat_count_history"][0]["total"] + 1e-10))
        
        return stats
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report across all adaptation cycles.
        
        Returns:
            Dictionary with summary report data
        """
        stats = self.get_summary_statistics()
        
        # Calculate additional metrics and insights
        summary = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "total_cycles": len(self.reports),
            "statistics": stats
        }
        
        # Identify most frequent adaptation types
        if stats["total_changes"] > 0:
            adaptation_ratios = {
                change_type: count / stats["total_changes"] 
                for change_type, count in stats["changes_by_type"].items()
                if count > 0
            }
            
            summary["adaptation_ratios"] = adaptation_ratios
            
            # Find most and least frequent types
            if adaptation_ratios:
                summary["most_frequent_adaptation"] = max(
                    adaptation_ratios.items(), key=lambda x: x[1]
                )
                summary["least_frequent_adaptation"] = min(
                    adaptation_ratios.items(), key=lambda x: x[1]
                )
        
        # Analyze splat count trends
        if "splat_count_history" in stats and len(stats["splat_count_history"]) >= 2:
            first = stats["splat_count_history"][0]
            last = stats["splat_count_history"][-1]
            
            # Calculate overall change
            summary["splat_count_change"] = {
                "initial": first["total"],
                "final": last["total"],
                "net_change": last["total"] - first["total"],
                "percent_change": (last["total"] - first["total"]) / first["total"] if first["total"] > 0 else 0
            }
            
            # Calculate changes by level
            level_changes = {}
            for level in all_levels:
                if level in first["by_level"] and level in last["by_level"]:
                    initial = first["by_level"][level]
                    final = last["by_level"][level]
                    
                    level_changes[level] = {
                        "initial": initial,
                        "final": final,
                        "net_change": final - initial,
                        "percent_change": (final - initial) / initial if initial > 0 else 0
                    }
            
            summary["level_count_changes"] = level_changes
        
        # Save summary report if requested
        if self.save_reports:
            filename = f"adaptation_summary_{self.session_id}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Make serializable
            summary_serializable = self._make_serializable(summary)
            
            try:
                with open(filepath, 'w') as f:
                    json.dump(summary_serializable, f, indent=2)
                logger.info(f"Saved adaptation summary to {filepath}")
            except Exception as e:
                logger.error(f"Error saving summary: {e}")
        
        return summary


class AdaptationVisualizer:
    """
    Visualizer for adaptation events and patterns.
    
    This class provides visualization tools for understanding adaptation
    operations and their effects on the splat registry.
    """
    
    def __init__(
        self, 
        output_dir: str = "hsa_adaptation_visualizations",
        save_visualizations: bool = True
    ):
        """
        Initialize the adaptation visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            save_visualizations: Whether to save visualizations to disk
        """
        self.output_dir = output_dir
        self.save_visualizations = save_visualizations
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if needed
        if self.save_visualizations:
            os.makedirs(output_dir, exist_ok=True)
    
    def visualize_adaptation_result(
        self,
        result: AdaptationResult,
        splat_registry: SplatRegistry,
        show: bool = True
    ) -> Optional[str]:
        """
        Create a visualization of an adaptation result.
        
        Args:
            result: The adaptation result to visualize
            splat_registry: The current splat registry
            show: Whether to display the visualization
            
        Returns:
            Path to the saved visualization if save_visualizations is True
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.gridspec import GridSpec
            
            # Create figure with subplots
            fig = plt.figure(figsize=(14, 10))
            gs = GridSpec(2, 2, figure=fig)
            
            # Extract data from result
            summary = result.get_summary()
            
            # Plot 1: Changes by type (pie chart)
            ax1 = fig.add_subplot(gs[0, 0])
            change_types = ["birth", "mitosis", "death", "merge", "adjust"]
            change_counts = [
                summary["birth_count"],
                summary["mitosis_count"],
                summary["death_count"], 
                summary["merge_count"],
                summary["adjust_count"]
            ]
            change_colors = ['green', 'blue', 'red', 'purple', 'orange']
            
            # Only plot non-zero values
            non_zero_types = [t for i, t in enumerate(change_types) if change_counts[i] > 0]
            non_zero_counts = [c for c in change_counts if c > 0]
            non_zero_colors = [c for i, c in enumerate(change_colors) if change_counts[i] > 0]
            
            if non_zero_counts:
                ax1.pie(
                    non_zero_counts, 
                    labels=non_zero_types,
                    colors=non_zero_colors,
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax1.set_title('Adaptation Types')
            else:
                ax1.text(0.5, 0.5, "No adaptations", ha='center', va='center')
                ax1.set_title('Adaptation Types (None)')
            
            # Plot 2: Splat counts before and after (bar chart)
            ax2 = fig.add_subplot(gs[0, 1])
            
            # Get levels and counts
            levels = list(summary["splats_by_level_before"].keys())
            counts_before = [summary["splats_by_level_before"].get(level, 0) for level in levels]
            counts_after = [summary["splats_by_level_after"].get(level, 0) for level in levels]
            
            # Calculate positions for grouped bars
            x = np.arange(len(levels))
            width = 0.35
            
            # Plot grouped bars
            ax2.bar(x - width/2, counts_before, width, label='Before', color='lightblue')
            ax2.bar(x + width/2, counts_after, width, label='After', color='lightgreen')
            
            # Add labels and legend
            ax2.set_xlabel('Hierarchy Level')
            ax2.set_ylabel('Splat Count')
            ax2.set_title('Splat Counts Before and After')
            ax2.set_xticks(x)
            ax2.set_xticklabels(levels)
            ax2.legend()
            
            # Plot 3: Net change by level (horizontal bar chart)
            ax3 = fig.add_subplot(gs[1, 0])
            
            # Calculate net change by level
            net_changes = [counts_after[i] - counts_before[i] for i in range(len(levels))]
            
            # Colors based on sign
            colors = ['green' if c > 0 else 'red' if c < 0 else 'gray' for c in net_changes]
            
            # Create horizontal bar chart
            bars = ax3.barh(levels, net_changes, color=colors)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                label_position = width + 0.5 if width > 0 else width - 0.5
                ax3.text(
                    label_position,
                    bar.get_y() + bar.get_height() / 2,
                    f'{int(width)}',
                    va='center',
                    ha='center' if -2 < width < 2 else ('left' if width < 0 else 'right')
                )
            
            # Add labels
            ax3.set_xlabel('Net Change')
            ax3.set_title('Net Change by Level')
            ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            # Plot 4: Adaptation timeline or summary stats
            ax4 = fig.add_subplot(gs[1, 1])
            
            # Create a text summary
            text_lines = [
                f"Total changes: {summary['total_changes']}",
                f"Duration: {summary['duration']:.2f} seconds",
                f"Total splats: {summary['splats_before']} → {summary['splats_after']}",
                f"Net change: {summary['splats_after'] - summary['splats_before']}",
                "",
                f"Births: {summary['birth_count']}",
                f"Mitosis: {summary['mitosis_count']}",
                f"Deaths: {summary['death_count']}",
                f"Merges: {summary['merge_count']}",
                f"Adjusts: {summary['adjust_count']}"
            ]
            
            # Add summary text
            ax4.text(
                0.5, 0.5, 
                '\n'.join(text_lines),
                ha='center', va='center',
                transform=ax4.transAxes
            )
            ax4.set_title('Adaptation Summary')
            ax4.axis('off')  # Hide axes
            
            # Add overall title
            fig.suptitle(
                f"Adaptation Result (ID: {self.session_id})",
                fontsize=16
            )
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save if requested
            if self.save_visualizations:
                filename = f"adaptation_result_{self.session_id}_{result.start_time:.0f}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Saved adaptation visualization to {filepath}")
                
                # Return the path to the saved file
                save_path = filepath
            else:
                save_path = None
            
            # Show if requested
            if show:
                plt.show()
            else:
                plt.close()
                
            return save_path
            
        except ImportError:
            logger.error("Matplotlib is required for visualization")
            return None
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
            return None
    
    def visualize_adaptation_history(
        self,
        reports: List[Dict[str, Any]],
        show: bool = True
    ) -> Optional[str]:
        """
        Create a visualization of adaptation history from multiple reports.
        
        Args:
            reports: List of adaptation reports
            show: Whether to display the visualization
            
        Returns:
            Path to the saved visualization if save_visualizations is True
        """
        if not reports:
            logger.warning("No reports to visualize")
            return None
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(2, 2, figure=fig)
            
            # Extract data from reports
            cycles = [report["cycle_id"] for report in reports]
            total_splats = [report["registry_state"]["total_splats"] for report in reports]
            
            # Get all unique levels across reports
            all_levels = set()
            for report in reports:
                for level in report["registry_state"]["splats_by_level"].keys():
                    all_levels.add(level)
            all_levels = sorted(list(all_levels))
            
            # Extract splat counts by level
            level_counts = {}
            for level in all_levels:
                level_counts[level] = [
                    report["registry_state"]["splats_by_level"].get(level, 0) 
                    for report in reports
                ]
            
            # Extract change counts by type
            birth_counts = [report["summary"].get("birth_count", 0) for report in reports]
            mitosis_counts = [report["summary"].get("mitosis_count", 0) for report in reports]
            death_counts = [report["summary"].get("death_count", 0) for report in reports]
            merge_counts = [report["summary"].get("merge_count", 0) for report in reports]
            adjust_counts = [report["summary"].get("adjust_count", 0) for report in reports]
            
            # Plot 1: Total splat count over time
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(cycles, total_splats, marker='o', color='blue', linewidth=2)
            ax1.set_xlabel('Adaptation Cycle')
            ax1.set_ylabel('Total Splats')
            ax1.set_title('Splat Count Over Time')
            ax1.grid(alpha=0.3)
            
            # Plot 2: Splat counts by level over time
            ax2 = fig.add_subplot(gs[0, 1])
            for level in all_levels:
                ax2.plot(cycles, level_counts[level], marker='o', label=level)
            ax2.set_xlabel('Adaptation Cycle')
            ax2.set_ylabel('Splat Count')
            ax2.set_title('Splat Counts by Level')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            # Plot 3: Adaptation counts by type over time
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(cycles, birth_counts, marker='o', color='green', label='Birth')
            ax3.plot(cycles, mitosis_counts, marker='s', color='blue', label='Mitosis')
            ax3.plot(cycles, death_counts, marker='x', color='red', label='Death')
            ax3.plot(cycles, merge_counts, marker='d', color='purple', label='Merge')
            ax3.plot(cycles, adjust_counts, marker='^', color='orange', label='Adjust')
            ax3.set_xlabel('Adaptation Cycle')
            ax3.set_ylabel('Count')
            ax3.set_title('Adaptation Counts by Type')
            ax3.legend()
            ax3.grid(alpha=0.3)
            
            # Plot 4: Net change per cycle
            ax4 = fig.add_subplot(gs[1, 1])
            
            # Calculate net change per cycle
            net_changes = []
            for i in range(1, len(reports)):
                prev = reports[i-1]["registry_state"]["total_splats"]
                curr = reports[i]["registry_state"]["total_splats"]
                net_changes.append(curr - prev)
            
            # Add a zero for the first cycle
            net_changes = [0] + net_changes
            
            # Create bar chart with colors based on sign
            bars = ax4.bar(
                cycles, 
                net_changes, 
                color=['green' if n > 0 else 'red' if n < 0 else 'gray' for n in net_changes]
            )
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                y_pos = height + 0.1 if height >= 0 else height - 0.6
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_pos,
                    f'{int(height)}',
                    ha='center'
                )
            
            ax4.set_xlabel('Adaptation Cycle')
            ax4.set_ylabel('Net Change')
            ax4.set_title('Net Change in Splat Count per Cycle')
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax4.grid(alpha=0.3)
            
            # Add overall title
            fig.suptitle(
                f"Adaptation History ({len(reports)} cycles, ID: {self.session_id})",
                fontsize=16
            )
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save if requested
            if self.save_visualizations:
                filename = f"adaptation_history_{self.session_id}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Saved adaptation history visualization to {filepath}")
                
                # Return the path to the saved file
                save_path = filepath
            else:
                save_path = None
            
            # Show if requested
            if show:
                plt.show()
            else:
                plt.close()
                
            return save_path
            
        except ImportError:
            logger.error("Matplotlib is required for visualization")
            return None
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
            return None
    
    def visualize_splat_changes(
        self,
        before_registry: SplatRegistry,
        after_registry: SplatRegistry,
        result: AdaptationResult,
        tokens: Optional[np.ndarray] = None,
        dim_reduction: bool = True,
        show: bool = True
    ) -> Optional[str]:
        """
        Visualize changes to splats between two registry states.
        
        Args:
            before_registry: Registry before adaptation
            after_registry: Registry after adaptation
            result: Adaptation result
            tokens: Optional token embeddings for context
            dim_reduction: Whether to use dimension reduction for visualization
            show: Whether to display the visualization
            
        Returns:
            Path to the saved visualization if save_visualizations is True
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
            
            # Get all levels across both registries
            all_levels = set(before_registry.hierarchy.levels)
            all_levels.update(after_registry.hierarchy.levels)
            all_levels = sorted(list(all_levels))
            
            # Prepare splat data for visualization
            before_splats = list(before_registry.splats.values())
            after_splats = list(after_registry.splats.values())
            
            # Extract positions
            before_positions = np.array([splat.position for splat in before_splats])
            after_positions = np.array([splat.position for splat in after_splats])
            
            # Apply dimension reduction if needed
            if dim_reduction and before_positions.shape[1] > 2:
                from sklearn.decomposition import PCA
                
                # Combine positions for consistent reduction
                all_positions = np.vstack([before_positions, after_positions])
                if tokens is not None:
                    all_data = np.vstack([all_positions, tokens])
                else:
                    all_data = all_positions
                
                # Apply PCA
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(all_data)
                
                # Split the reduced data back
                before_positions_2d = reduced_data[:len(before_positions)]
                after_positions_2d = reduced_data[len(before_positions):len(before_positions) + len(after_positions)]
                
                if tokens is not None:
                    tokens_2d = reduced_data[len(before_positions) + len(after_positions):]
                else:
                    tokens_2d = None
            else:
                # Use first two dimensions directly
                before_positions_2d = before_positions[:, :2]
                after_positions_2d = after_positions[:, :2]
                tokens_2d = tokens[:, :2] if tokens is not None else None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot tokens if available
            if tokens_2d is not None:
                ax.scatter(
                    tokens_2d[:, 0], 
                    tokens_2d[:, 1], 
                    color='lightgray', 
                    alpha=0.2,
                    marker='.',
                    label='Tokens'
                )
            
            # Plot "before" splats
            for i, splat in enumerate(before_splats):
                # Get level for color
                level = splat.level
                level_idx = all_levels.index(level) if level in all_levels else 0
                
                # Create color with varying hue based on level
                color = plt.cm.tab10(level_idx % 10)
                
                # Plot position
                ax.scatter(
                    before_positions_2d[i, 0],
                    before_positions_2d[i, 1],
                    color=color,
                    alpha=0.5,
                    marker='o',
                    s=50,
                    edgecolor='black'
                )
                
                # Add text label
                ax.text(
                    before_positions_2d[i, 0],
                    before_positions_2d[i, 1] + 0.1,
                    f"{splat.id[-4:]}",
                    fontsize=8,
                    ha='center'
                )
            
            # Plot "after" splats
            for i, splat in enumerate(after_splats):
                # Get level for color
                level = splat.level
                level_idx = all_levels.index(level) if level in all_levels else 0
                
                # Create color with varying hue based on level
                color = plt.cm.tab10(level_idx % 10)
                
                # Plot position
                ax.scatter(
                    after_positions_2d[i, 0],
                    after_positions_2d[i, 1],
                    color=color,
                    alpha=0.9,
                    marker='^',
                    s=50,
                    edgecolor='black'
                )
                
                # Add text label
                ax.text(
                    after_positions_2d[i, 0],
                    after_positions_2d[i, 1] - 0.2,
                    f"{splat.id[-4:]}",
                    fontsize=8,
                    ha='center'
                )
            
            # Find birth and death operations to visualize
            births = [c for c in result.changes if c.change_type == AdaptationType.BIRTH]
            deaths = [c for c in result.changes if c.change_type == AdaptationType.DEATH]
            
            # Highlight births with green arrows
            for birth in births:
                # Find the splat in after_registry
                for i, splat in enumerate(after_splats):
                    if splat.id == birth.splat_id:
                        # Highlight with a green circle
                        ax.scatter(
                            after_positions_2d[i, 0],
                            after_positions_2d[i, 1],
                            color='none',
                            marker='o',
                            s=100,
                            edgecolor='green',
                            linewidth=2
                        )
                        break
            
            # Highlight deaths with red Xs
            for death in deaths:
                # Find the splat in before_registry
                for i, splat in enumerate(before_splats):
                    if splat.id == death.splat_id:
                        # Highlight with a red X
                        ax.scatter(
                            before_positions_2d[i, 0],
                            before_positions_2d[i, 1],
                            color='red',
                            marker='x',
                            s=100,
                            linewidth=2
                        )
                        break
            
            # Add legend with levels
            legend_elements = []
            for level in all_levels:
                level_idx = all_levels.index(level)
                color = plt.cm.tab10(level_idx % 10)
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', label=level,
                           markerfacecolor=color, markersize=10)
                )
            
            # Add markers for before/after
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', label='Before',
                       markerfacecolor='gray', markersize=10)
            )
            legend_elements.append(
                plt.Line2D([0], [0], marker='^', color='w', label='After',
                       markerfacecolor='gray', markersize=10)
            )
            
            # Add markers for birth/death
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', label='Birth',
                       markerfacecolor='none', markeredgecolor='green', 
                       markersize=10, markeredgewidth=2)
            )
            legend_elements.append(
                plt.Line2D([0], [0], marker='x', color='red', label='Death',
                       markersize=10, linewidth=2)
            )
            
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Add title and labels
            ax.set_title(f"Splat Changes (ID: {self.session_id})")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            
            # Add summary text
            summary_text = [
                f"Total splats: {len(before_splats)} → {len(after_splats)}",
                f"Births: {result.birth_count}",
                f"Deaths: {result.death_count}",
                f"Mitosis: {result.mitosis_count}",
                f"Merges: {result.merge_count}",
                f"Adjusts: {result.adjust_count}"
            ]
            
            # Add text box with summary
            plt.annotate(
                '\n'.join(summary_text),
                xy=(0.02, 0.02),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
            )
            
            # Save if requested
            if self.save_visualizations:
                filename = f"splat_changes_{self.session_id}_{result.start_time:.0f}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Saved splat changes visualization to {filepath}")
                
                # Return the path to the saved file
                save_path = filepath
            else:
                save_path = None
            
            # Show if requested
            if show:
                plt.show()
            else:
                plt.close()
                
            return save_path
            
        except ImportError:
            logger.error("Matplotlib and scikit-learn are required for visualization")
            return None
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
            return None


# Create default reporter and visualizer
default_reporter = AdaptationReporter()
default_visualizer = AdaptationVisualizer()


def generate_report(
    result: AdaptationResult,
    splat_registry: SplatRegistry,
    metrics: Optional[Dict[str, Any]] = None,
    reporter: Optional[AdaptationReporter] = None
) -> Dict[str, Any]:
    """
    Generate a report for an adaptation result.
    
    Args:
        result: The adaptation result to report on
        splat_registry: The current splat registry
        metrics: Optional additional metrics to include
        reporter: Optional reporter to use
        
    Returns:
        Report data
    """
    # Use default reporter if none provided
    if reporter is None:
        reporter = default_reporter
    
    return reporter.generate_report(result, splat_registry, metrics)


def visualize_result(
    result: AdaptationResult,
    splat_registry: SplatRegistry,
    show: bool = True,
    visualizer: Optional[AdaptationVisualizer] = None
) -> Optional[str]:
    """
    Visualize an adaptation result.
    
    Args:
        result: The adaptation result to visualize
        splat_registry: The current splat registry
        show: Whether to display the visualization
        visualizer: Optional visualizer to use
        
    Returns:
        Path to the saved visualization if successful
    """
    # Use default visualizer if none provided
    if visualizer is None:
        visualizer = default_visualizer
    
    return visualizer.visualize_adaptation_result(result, splat_registry, show)


def visualize_history(
    reports: List[Dict[str, Any]],
    show: bool = True,
    visualizer: Optional[AdaptationVisualizer] = None
) -> Optional[str]:
    """
    Visualize adaptation history from reports.
    
    Args:
        reports: List of adaptation reports
        show: Whether to display the visualization
        visualizer: Optional visualizer to use
        
    Returns:
        Path to the saved visualization if successful
    """
    # Use default visualizer if none provided
    if visualizer is None:
        visualizer = default_visualizer
    
    return visualizer.visualize_adaptation_history(reports, show)


def visualize_splat_changes(
    before_registry: SplatRegistry,
    after_registry: SplatRegistry,
    result: AdaptationResult,
    tokens: Optional[np.ndarray] = None,
    dim_reduction: bool = True,
    show: bool = True,
    visualizer: Optional[AdaptationVisualizer] = None
) -> Optional[str]:
    """
    Visualize changes to splats between two registry states.
    
    Args:
        before_registry: Registry before adaptation
        after_registry: Registry after adaptation
        result: Adaptation result
        tokens: Optional token embeddings for context
        dim_reduction: Whether to use dimension reduction for visualization
        show: Whether to display the visualization
        visualizer: Optional visualizer to use
        
    Returns:
        Path to the saved visualization if successful
    """
    # Use default visualizer if none provided
    if visualizer is None:
        visualizer = default_visualizer
    
    return visualizer.visualize_splat_changes(
        before_registry, after_registry, result, tokens, dim_reduction, show
    )
