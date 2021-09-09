"""Graph2Vec module."""
import time
import json
import glob
import hashlib
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from param_parser import parameter_parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import readlabel
import pickle

import os

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(k)[:13] for k, v in features]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            #print(node)
            #time.sleep(100000)
            nebs = self.graph.neighbors(node)   
            #print('Nebs:')
            features=[]                         
            features.append(node)
            for n in nebs:
                features.append(n)
                #print(n)
            
            features = "_".join(features)       
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing        #hashing for relabeling 
            
        self.extracted_features = self.extracted_features + list(new_features.values())
        #self.extracted_features = list(new_features.values())    
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            #print('iterations:',_)
            self.features = self.do_a_recursion()

def build_graph(path):
    
    with open(path, 'r') as myfile:
        data=myfile.read()
    #print(data)
    
    label={}
    G=nx.DiGraph()
    for lines in data.split('\n'):
        tmp=[]
        for words in lines.split():
            #print(words)
            if words[0]=='"':
                words=words.replace('"','')
            tmp.append(words)
       
        try:
            if tmp[1][1]=='l':
                func=tmp[1][7:]
                func=func.replace('"','')
                #print(func)
                label[tmp[0]]=func
        except:
            pass
       
    for lines in data.split('\n'):
        tmp=[]
        for words in lines.split():
            #print(words)
            if words[0]=='"':
                words=words.replace('"','')
            tmp.append(words)
       
        try:
            if tmp[1]=='->':
                #print(label[tmp[0]],label[tmp[2]])     
                G.add_edge(label[tmp[0]],label[tmp[2]])     
        except:
            pass

    return G

def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path.strip(".dot").split("/")[-1]

    graph=build_graph(path)

    #features = list(graph.nodes)
    
    features = nx.degree(graph)
    #features = {int(k): v for k, v in features}
    return graph, features, name 

def feature_extractor(path, rounds):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])

    return doc

def save_embedding(output_path, model, files, dimensions, path):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    label=readlabel.read_label()
    
    with open("label.pkl", 'wb') as f:
        pickle.dump(label, f)
    
    for f in files:
        #modified file name
        f=f.replace('modified ','')
        f=f.replace(path,'')
        f=f.replace('.dot','')
        #log_file.append(f)
        
        identifier = f.split("/")[-1].strip(".dot")
        try:
            out.append([f] + list(model.docvecs["g_"+identifier])+[label[f]])
            #out.append([f] + list(model.docvecs["g_"+identifier]))
            
            
        except:
            pass
    column_names = ["type"]+["x_"+str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(["type"])
    #print(out)
    out.to_csv(output_path, mode='w',index=None,header=False)

    

def main(args):
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """
    graphs = glob.glob(args.input_path + "*.dot")
    #print(graphs)
    print("\nFeature extraction started.\n")
    document_collections = Parallel(n_jobs=args.workers)(delayed(feature_extractor)(g, args.wl_iterations) for g in tqdm(graphs))
    print("\nOptimization started.\n")

    #print(document_collections)

    
    model = Doc2Vec(document_collections, 
                    vector_size=args.dimensions, 
                    window=0, 
                    min_count=args.min_count, 
                    dm=0, 
                    sample=args.down_sampling, 
                    workers=args.workers, 
                    epochs=args.epochs, 
                    alpha=args.learning_rate)

    save_embedding(args.output_path, model, graphs, args.dimensions,args.input_path)
    

if __name__ == "__main__":
    start = time.time()
    args = parameter_parser()
    main(args)
    end = time.time()

    print(f'spend {end - start} seconds')

