#!/usr/bin/env python3
"""v4.2 entry point; delegates to the frozen trainer with v4.2 feature lists."""
import sys
import feature_sets_v4_2
sys.modules['feature_sets_v4']=feature_sets_v4_2
from train_v4_roofline_xgboost import main
if __name__=='__main__':main()
