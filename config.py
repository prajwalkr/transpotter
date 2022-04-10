import configargparse

list_of_phones = ["AH0", "N", "S", "L", "T", "R", 
      "K", "D", "IH0", "M", "Z", "ER0", "IY0", "B", "EH1", "P", "AA1", "AE1", "IH1", 
      "F", "G", "V", "IY1", "NG", "HH", "EY1", "W", "SH", "OW1", "OW0", "AO1", "AY1", 
      "AH1", "UW1", "JH", "Y", "AA0", "CH", "ER1", "IH2", "EH2", "EY2", "AE2", "AY2", 
      "AA2", "TH", "EH0", "IY2", "OW2", "AW1", "UW0", "AO2", "AE0", "UH1", "AO0", 
      "AY0", "UW2", "AH2", "EY0", "OY1", "AW2", "ER2", "DH", "ZH", "UH2", "AW0", 
      "UH0", "OY2", "OY0"]

def load_args():

  parser = configargparse.ArgumentParser(description = "main")
  parser.add_argument('--device', type=str, default='cuda')
  parser.add_argument('--mode', type=str, default='test')

  parser.add_argument('--ckpt_path', default=None)

  parser.add_argument('--thresh', default=3, type=int, 
                  help='min number of phonemes in a word at test time')

  # Data
  parser.add_argument('--data_root', type=str, required=True, help='Path to the folder with the features')

  parser.add_argument('--test_pkl_file', type=str, help='Path to the pkl of test/val samples')

  # Transformer config
  parser.add_argument('--num_blocks', type=int, default=6, help='# of transformer blocks')
  parser.add_argument('--hidden_units', type=int, default=512, help='Transformer model size')
  parser.add_argument('--num_heads', type=int, default=8, help='# of attention heads')
  parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout probability')

  # inference params
  parser.add_argument('--localization', action='store_true', help='Test with localization')
  parser.add_argument('--loc_thresh', type=float, default=0.5, 
                                            help='Threshold on binary frame score')
  parser.add_argument('--iou_thresh', type=float, default=0.5, 
                                            help='Min. IOU to treat as correct loc.')
  args = parser.parse_args()

  return args

