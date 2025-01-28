#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch

import deep_sdf
import deep_sdf.workspace as ws

def load_latent_vector(filename):
    if not os.path.isfile(filename):
        raise Exception('latent state file "{}" does not exist'.format(filename))

    data = torch.load(filename).squeeze()
    return data

def interpolate(latent_code_1, latent_code_2, num_interpolation_steps):
    interpolated_codes = []
    for i in range(0, num_interpolation_steps + 1):
        interpolated_code = latent_code_1 + i*(latent_code_2 - latent_code_1)/num_interpolation_steps
        interpolated_codes.append(interpolated_code)

    return interpolated_codes

def interpolate_embeddings(embedding_1, embedding_2, num_interpolation_steps):
    interpolated_embeddings = []
    for i in range(0, num_interpolation_steps + 1):
        interpolated_embedding = embedding_1 + i*(embedding_2 - embedding_1)/num_interpolation_steps
        interpolated_embeddings.append(interpolated_embedding)
    return interpolated_embeddings

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    
    arg_parser.add_argument(
        "--num_ipo",
        "-n",
        dest="num_ipo",
        required=True,
        default="5",
        help="Number of interpolation steps.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()
        
    deep_sdf.configure_logging(args)
    
    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))
            
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    npz_filenames, class_names = deep_sdf.data.get_instance_classnames_filenames(args.data_source, split)

    logging.debug(decoder)

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    if len(npz_filenames) < 2:
        raise Exception(
            'Provide at least two shapes\' names to conduct latentspace interpolation!'
        )
        
    for ii, npz in enumerate(npz_filenames[:-1]):
        if "npz" not in npz:
            continue

        class_embedding_0 = None
        if specs["NetworkSpecs"]["class_embedding"]:
            class_embedding_0 = specs["ClassEmbedding"][class_names[ii]]

        class_embedding_1 = None
        if specs["NetworkSpecs"]["class_embedding"]:
            class_embedding_1 = specs["ClassEmbedding"][class_names[ii+1]]

        embedding_vec_0 = torch.zeros((9))
        embedding_vec_0[class_embedding_0] = 1
                
        embedding_vec_1 = torch.zeros((9))
        embedding_vec_1[class_embedding_1] = 1
        
        mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4]+"_interpolated")
        latent_0_filename = os.path.join(
            reconstruction_codes_dir, npz[:-4] + ".pth"
        )
        
        latent_1_filename = os.path.join(
            reconstruction_codes_dir, npz_filenames[ii+1][:-4] + ".pth"
        )

        latent_0 = load_latent_vector(latent_0_filename )
        latent_1 = load_latent_vector(latent_1_filename )

        interpolated_latent_vecs = interpolate(latent_0, latent_1, int(args.num_ipo)-1)
        interpolated_embeddings = interpolate_embeddings(embedding_vec_0, embedding_vec_1, int(args.num_ipo)-1)
        decoder.eval()

        if not os.path.exists(os.path.dirname(mesh_filename)):
            os.makedirs(os.path.dirname(mesh_filename))
        
        start = time.time()
        for i, latent_vec in enumerate(interpolated_latent_vecs):
            with torch.no_grad():
                deep_sdf.mesh.create_mesh(
                    decoder, latent_vec, mesh_filename+f"-{i}", N=256, max_batch=int(2 ** 18), class_embedding=interpolated_embeddings[i]
                )
        logging.debug("total time: {}".format(time.time() - start))
