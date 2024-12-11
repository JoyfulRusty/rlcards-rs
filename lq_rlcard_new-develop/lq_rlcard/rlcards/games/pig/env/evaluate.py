import os
import argparse

from rlcards.games.pig.evaluation.simulation import evaluate

BASE_PATH = 'D:\\lq_rlcard\\rlcards\\predict\\pig'
NEW_MODEL_PATH = os.path.join(os.path.dirname(BASE_PATH), "pig", "models")

# 权重
NEW_WEIGHTS = 832806400
NEW_MODEL_PATH_1 = os.path.join(NEW_MODEL_PATH, f"landlord1_weights_{NEW_WEIGHTS}.ckpt")
NEW_MODEL_PATH_2 = os.path.join(NEW_MODEL_PATH, f"landlord2_weights_{NEW_WEIGHTS}.ckpt")
NEW_MODEL_PATH_3 = os.path.join(NEW_MODEL_PATH, f"landlord3_weights_{NEW_WEIGHTS}.ckpt")
NEW_MODEL_PATH_4 = os.path.join(NEW_MODEL_PATH, f"landlord4_weights_{NEW_WEIGHTS}.ckpt")

OLD_MODEL_PATH = os.path.join(os.path.dirname(BASE_PATH), "pig", "models")
# 权重
OLD_WEIGHTS = 1033779200
OLD_MODEL_PATH_1 = os.path.join(OLD_MODEL_PATH, f"landlord1_weights_{OLD_WEIGHTS}.ckpt")
OLD_MODEL_PATH_2 = os.path.join(OLD_MODEL_PATH, f"landlord2_weights_{OLD_WEIGHTS}.ckpt")
OLD_MODEL_PATH_3 = os.path.join(OLD_MODEL_PATH, f"landlord3_weights_{OLD_WEIGHTS}.ckpt")
OLD_MODEL_PATH_4 = os.path.join(OLD_MODEL_PATH, f"landlord4_weights_{OLD_WEIGHTS}.ckpt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Dou Dizhu Evaluation')
    parser.add_argument('--landlord1', type=str,
                        default=str(OLD_MODEL_PATH_1))
    parser.add_argument('--landlord2', type=str,
                        default=str(OLD_MODEL_PATH_2))
    parser.add_argument('--landlord3', type=str,
                        default=str(NEW_MODEL_PATH_3))
    parser.add_argument('--landlord4', type=str,
                        default=str(NEW_MODEL_PATH_4))
    parser.add_argument('--eval_data', type=str,
                        default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    evaluate(args.landlord1,
             args.landlord2,
             args.landlord3,
             args.landlord4,
             args.eval_data,
             args.num_workers)
