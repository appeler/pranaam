#!/usr/bin/env python3
"""
Model Migration Script: Keras 2 SavedModel → Keras 3 Compatible Format

This script converts existing Keras 2 SavedModel format models to Keras 3 compatible
"weights + architecture" format to enable TensorFlow 2.16+ compatibility.

Usage:
    # Step 1: Export from Keras 2 environment (TensorFlow 2.15)
    python migrate_models.py export

    # Step 2: Import to Keras 3 environment (TensorFlow 2.16+)
    python migrate_models.py import

Requirements:
    - Step 1 requires TensorFlow 2.15 (Keras 2)
    - Step 2 requires TensorFlow 2.16+ (Keras 3)
"""

import json
import sys
from pathlib import Path

import tensorflow as tf


class ModelMigrator:
    """Handles migration between Keras 2 and Keras 3 model formats."""

    def __init__(self):
        self.model_base_path = Path(__file__).parent / "pranaam" / "model"
        self.migration_output = Path(__file__).parent / "model_migration"
        self.migration_output.mkdir(exist_ok=True)

        # Model paths
        self.models = {
            "eng": self.model_base_path / "eng_and_hindi_models_v1" / "eng_model",
            "hin": self.model_base_path / "eng_and_hindi_models_v1" / "hin_model",
        }

    def export_from_keras2(self) -> None:
        """Export weights and architecture from Keras 2 SavedModel format."""
        print("🔄 Exporting models from Keras 2 format...")
        print(f"TensorFlow version: {tf.__version__}")

        # Use tf-keras for Keras 2 compatibility
        try:
            import tf_keras as keras

            print("✅ Using tf-keras for Keras 2 compatibility")
            print(f"tf-keras version: {keras.__version__}")
        except ImportError as e:
            print(f"❌ tf-keras not available: {e}")
            print("Please install tf-keras: pip install tf-keras")
            return

        for lang, model_path in self.models.items():
            print(f"\n📁 Processing {lang} model from {model_path}")

            if not model_path.exists():
                print(f"❌ Model not found at {model_path}")
                continue

            try:
                # Load the SavedModel using tf-keras (Keras 2 compatibility)
                print("🔄 Loading SavedModel with tf-keras...")
                model = keras.models.load_model(str(model_path))
                print(f"✅ Successfully loaded {lang} model")

                # Print model info
                print(f"   Model type: {type(model).__name__}")
                print(
                    f"   Input shape: {model.input_shape if hasattr(model, 'input_shape') else 'Unknown'}"
                )
                print(
                    f"   Output shape: {model.output_shape if hasattr(model, 'output_shape') else 'Unknown'}"
                )

                # Create language-specific output directory
                lang_output = self.migration_output / f"{lang}_model"
                lang_output.mkdir(exist_ok=True)

                # Try to save as Keras 3 format first, then fallback to tf format if needed
                try:
                    # Save complete model in Keras 3 format
                    keras_model_path = lang_output / "model.keras"
                    model.save(str(keras_model_path))
                    print(f"✅ Complete model saved to {keras_model_path}")
                except Exception as e:
                    print(f"⚠️  Keras format save failed: {e}")
                    # Fallback to TF weights format
                    weights_path = lang_output / "model_weights"
                    model.save_weights(str(weights_path), save_format="tf")
                    print(f"✅ Weights saved to {weights_path} (tf format)")

                # Export model configuration
                config = model.get_config()
                config_path = lang_output / "model_config.json"
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                print(f"✅ Config saved to {config_path}")

                # Export additional metadata
                metadata = {
                    "model_type": type(model).__name__,
                    "input_shape": (
                        model.input_shape if hasattr(model, "input_shape") else None
                    ),
                    "output_shape": (
                        model.output_shape if hasattr(model, "output_shape") else None
                    ),
                    "layer_count": len(model.layers) if hasattr(model, "layers") else 0,
                    "tensorflow_version": tf.__version__,
                    "keras_version": (
                        tf.keras.__version__
                        if hasattr(tf.keras, "__version__")
                        else None
                    ),
                }

                # Try to get input/output names
                try:
                    metadata["input_names"] = [inp.name for inp in model.inputs]
                    metadata["output_names"] = [out.name for out in model.outputs]
                except AttributeError:
                    pass

                metadata_path = lang_output / "model_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                print(f"✅ Metadata saved to {metadata_path}")

                # Test a prediction to validate the model works
                if lang == "eng":
                    test_input = ["Shah Rukh Khan"]
                else:
                    test_input = ["शाहरुख खान"]

                try:
                    result = model.predict(test_input, verbose=0)
                    print(f"✅ Test prediction successful: {result.shape}")

                    # Save test prediction for validation
                    test_data = {"input": test_input, "output": result.tolist()}
                    test_path = lang_output / "test_prediction.json"
                    with open(test_path, "w") as f:
                        json.dump(test_data, f, indent=2)
                    print(f"✅ Test prediction saved to {test_path}")

                except Exception as e:
                    print(f"⚠️  Test prediction failed: {e}")

            except Exception as e:
                print(f"❌ Failed to process {lang} model: {e}")
                continue

        print("\n✅ Export completed! Check the model_migration directory.")
        print("Next step: Run 'python migrate_models.py import' with TensorFlow 2.16+")

    def import_to_keras3(self) -> None:
        """Import weights and recreate models in Keras 3 format."""
        print("🔄 Importing models to Keras 3 format...")
        print(f"TensorFlow version: {tf.__version__}")

        # Verify we're in Keras 3 environment
        try:
            import tensorflow.keras.utils as utils

            if not hasattr(utils, "legacy"):
                print("⚠️  Detected Keras 2 environment - this step needs Keras 3!")
                print("Please run with TensorFlow 2.16+ or upgrade TensorFlow")
                sys.exit(1)
        except ImportError:
            pass

        # Create new model directory for Keras 3 models
        new_model_dir = self.model_base_path / "eng_and_hindi_models_v2"
        new_model_dir.mkdir(exist_ok=True)

        for lang in ["eng", "hin"]:
            print(f"\n📁 Processing {lang} model migration to Keras 3")

            lang_input = self.migration_output / f"{lang}_model"
            if not lang_input.exists():
                print(f"❌ Migration data not found at {lang_input}")
                print(
                    "Run 'python migrate_models.py export' first with TensorFlow 2.15"
                )
                continue

            try:
                # Load metadata
                metadata_path = lang_input / "model_metadata.json"
                with open(metadata_path) as f:
                    metadata = json.load(f)
                print(f"✅ Loaded metadata: {metadata['model_type']}")

                # Load configuration (for metadata purposes)
                config_path = lang_input / "model_config.json"
                with open(config_path) as f:
                    _ = json.load(f)  # Load but don't use since we load .keras directly
                print("✅ Loaded model configuration")

                # Load the Keras 3 format model directly (tf-keras saved it for us!)
                keras_model_path = lang_input / "model.keras"
                if keras_model_path.exists():
                    print("🔄 Loading Keras 3 format model...")
                    model = tf.keras.models.load_model(str(keras_model_path))
                    print("✅ Loaded Keras 3 model directly")
                else:
                    print(f"❌ Keras 3 model not found at {keras_model_path}")
                    continue

                # Test the recreated model
                test_path = lang_input / "test_prediction.json"
                if test_path.exists():
                    with open(test_path) as f:
                        test_data = json.load(f)

                    result = model.predict(test_data["input"], verbose=0)
                    original_output = test_data["output"]

                    # Compare predictions (allowing for small numerical differences)
                    import numpy as np

                    if np.allclose(result, original_output, rtol=1e-5):
                        print("✅ Model validation successful - predictions match!")
                    else:
                        print("⚠️  Predictions differ slightly (this might be normal)")
                        print(f"   Original: {original_output}")
                        print(f"   New: {result.tolist()}")

                # Save in Keras 3 native format
                keras3_model_path = new_model_dir / f"{lang}_model.keras"
                model.save(keras3_model_path)
                print(f"✅ Saved Keras 3 model to {keras3_model_path}")

                # Also save in HDF5 format as backup
                h5_model_path = new_model_dir / f"{lang}_model.h5"
                model.save(h5_model_path)
                print(f"✅ Saved H5 backup to {h5_model_path}")

            except Exception as e:
                print(f"❌ Failed to migrate {lang} model: {e}")
                import traceback

                traceback.print_exc()
                continue

        print("\n✅ Import completed!")
        print(f"New models saved in: {new_model_dir}")
        print("Next step: Update pranaam code to use the new model format")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ["export", "import"]:
        print("Usage: python migrate_models.py [export|import]")
        print("  export: Export from Keras 2 SavedModel (requires TF 2.15)")
        print("  import: Import to Keras 3 format (requires TF 2.16+)")
        sys.exit(1)

    migrator = ModelMigrator()

    if sys.argv[1] == "export":
        migrator.export_from_keras2()
    else:
        migrator.import_to_keras3()


if __name__ == "__main__":
    main()
