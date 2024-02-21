from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

from src.protos.python.feature_extraction import feature_extraction_pb2_grpc, feature_extraction_pb2
from dataset_processing import make_inference_data_loader, make_data_loader
from model import *
import src.utils.utils as util

# =============================================================================

import grpc
import pickle
from concurrent import futures
import argparse
import socket
from pytorch_lightning.trainer.trainer import Trainer
import torch


class FeatureExtractionServicer(feature_extraction_pb2_grpc.FeatureExtractorServicer):

    def __init__(
            self, 
            **kwargs
    ):
        cfg = kwargs["config"]
        pretrained_num_classes = kwargs["pretrained_num_classes"]
        self.device = kwargs["device"]

        self.model = build_model(cfg, pretrained_num_classes)
        self.model.to(self.device)
        self.model.eval()

        self.n_called = 0


    def predict(self, request, context):
        img_batch = pickle.loads(request.imgs_pkl)

        with torch.no_grad():
            self.n_called += 1
            img_batch = img_batch.to(self.device)

            print(f'[{self.n_called}th runs] Processing batch of size', img_batch.size(0))
            features = self.model(img_batch).cpu()

        features_pkl = pickle.dumps(features)
        return feature_extraction_pb2.FeatureBatch(features_pkl=features_pkl)


def parse_kwargs():
    
    ap = argparse.ArgumentParser()

    ap.add_argument("-c", "--config", default="DEFAULT", help="configuration file to use")
    ap.add_argument("--port", required=True, type=int, help="Port number")
    ap.add_argument("--device", required=True, type=int)
    ap.add_argument("--pretrained_num_classes", required=True, type=int)

    kwargs = vars(ap.parse_args())

    return kwargs


if __name__ == '__main__':

    kwargs = parse_kwargs()
    cfg = util.load_defaults(["configs/dataset_AIC.yaml", "configs/baseline.yaml"])
    kwargs["config"] = cfg

    port = kwargs["port"]

    # check if the port is not used
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('localhost', port)) == 0:
            raise ValueError(f"Port {port} is already in use. Please check if the server is already running.")
        
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    feature_extraction_pb2_grpc.add_FeatureExtractorServicer_to_server(FeatureExtractionServicer(**kwargs), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Server started at port {port}")
    server.wait_for_termination()
