#!/bin/bash
# Comprehensive Enhanced Dataset Validation Script

echo "=========================================="
echo "Comprehensive Enhanced Dataset Validation"
echo "=========================================="

# Get the project directory
PROJECT_DIR="/gpfs/wolf2/cades/cli185/proj-shared/guzhuowei0407/training_data_generation"

# Activate Python virtual environment
source ${PROJECT_DIR}/venv_py311/bin/activate

echo "Running comprehensive validation of enhanced dataset..."
echo ""
echo "This validation includes:"
echo "1. ✅ Forcing data monthly averaging verification (240 values for 20 years)"
echo "2. ✅ Data existence and format validation"
echo "3. ✅ Pool data consistency with restart file (spatial mapping)"
echo "4. ✅ PFT data consistency with CLM parameters (value-by-value)"
echo "5. ✅ Forcing data consistency and scientific validity"
echo "6. ✅ History vs restart file comparison"
echo "7. ✅ Gridcell-by-gridcell validation (first 5000 gridcells)"
echo "8. ✅ Spatial mapping verification"
echo ""

# Run the comprehensive validation script
python ${PROJECT_DIR}/validation/comprehensive_validation.py

# Capture exit code
validation_result=$?

echo ""
echo "=========================================="
if [ $validation_result -eq 0 ]; then
    echo "🎉 COMPREHENSIVE VALIDATION COMPLETED SUCCESSFULLY!"
    echo "✅ Enhanced dataset is completely validated and ready for use"
    echo ""
    echo "Validation Summary:"
    echo "  - Monthly averaging: ✅ Verified (240 values for 20 years)"
    echo "  - Data existence: ✅ Verified"
    echo "  - Pool data consistency: ✅ Verified (spatial mapping)"
    echo "  - PFT data consistency: ✅ Verified (value-by-value)"
    echo "  - Forcing data consistency: ✅ Verified"
    echo "  - History/Restart comparison: ✅ Verified"
    echo "  - Gridcell mapping: ✅ Verified (5000 gridcells)"
    echo "  - Data integrity: ✅ Verified"
    echo ""
    echo "Your enhanced dataset is scientifically accurate and ready for machine learning!"
else
    echo "❌ COMPREHENSIVE VALIDATION FAILED!"
    echo "⚠️  Enhanced dataset needs review"
    echo ""
    echo "Please check the validation output above for details."
    echo "Some validations may have failed and need attention."
fi
echo "=========================================="

exit $validation_result

