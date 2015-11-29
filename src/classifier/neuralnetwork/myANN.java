/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author Rakhmatullah Yoga S, Linda Sekawati, Joshua Bezaleel Abednego
 */
public class myANN extends Classifier {
    private Node[][] network;
    private Attribute attrClass;
    private double threshold;
    private double givenWeight;
    private double target;
    private double learningRate;
    private double momentum;
    private int nbLayer;
    private int[] nbNeuron;
    private int nbInput;
    private int maxEpoch;
    private boolean singlePerceptron;
    private boolean randomWeight;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        NominalToBinary nomToBinFilter = new NominalToBinary();
        Normalize normalizeFilter = new Normalize();
        
        // PREPARE DATASET
        data = new Instances(data);
        data.deleteWithMissingClass();
        nomToBinFilter.setInputFormat(data);
        data = Filter.useFilter(data, nomToBinFilter);
        normalizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, normalizeFilter);
        
        // BUILDING NETWORK
        nbInput = data.numAttributes()-1;
        nbNeuron = new int[nbLayer+1];
        // WARNING: assign dulu berapa neuron per layernya
        int i=0;
        for(i=0; i<nbLayer; i++) {
            nbNeuron[i] = nbLayer-i;
        }
        if(data.classAttribute().isNumeric())
            nbNeuron[i] = 1;
        else
            nbNeuron[i] = data.classAttribute().numValues();
        network = new Node[nbLayer][];
        for(i=0; i<nbNeuron.length; i++) {
            network[i] = new Node[nbNeuron[i]];
            for(int j=0; j<nbNeuron[i]; j++) {
                network[i][j] = new Node();
                if(i==0) {
                    network[i][j].initWeight(nbInput+1, givenWeight);
                }
                else if(i==nbLayer-1) {
                    if(data.classAttribute().isNumeric())
                        network[i][j].setSigmoid(false);
                    network[i][j].initWeight(nbNeuron[i-1]+1, givenWeight);
                }
                else {
                    network[i][j].initWeight(nbNeuron[i-1]+1, givenWeight);
                }
                if(randomWeight)
                    network[i][j].randomizeWeight();
            }
        }
        // END OF BUILDING NETWORK
        
        // START CLASSIFYING
        attrClass = data.classAttribute();
        int epoch=0;
        double MSE = Double.POSITIVE_INFINITY;
        while(epoch<maxEpoch && MSE>threshold) {
            MSE = 0.0;
            double[] targets = null;
            for (int in=0; in<data.numInstances(); in++) {
                Instance instance = data.instance(in);
                if(instance.classAttribute().isNominal()) {
                    targets = new double[attrClass.numValues()];
                    for (double targetN : targets) {
                        targetN = 0.0;
                    }
                    targets[(int)instance.classValue()] = 1.0;
                }
                else {
                    targets = new double[1];
                    targets[0] = instance.classValue();
                }
                double[] inputAttr = new double[nbInput];
                int idx = 0;
                for (double attr : inputAttr) {
                    attr = instance.value(idx);
                    idx++;
                }
                // END OF BUILDING NETWORK
                // ==============================
                // operasi feed forward
                double[] output;                        // output masing2 class layer
                double[] localInput = inputAttr;
                for(int level=0; level<network.length; level++) {
                    double[] result = new double[network[level].length];
                    for(int neuron=0; neuron<network[level].length; neuron++) {
                        network[level][neuron].setInput(localInput);
                        network[level][neuron].computeValue();
                        result[neuron] = network[level][neuron].getValue();
                    }
                    localInput = result;
                }
                output = localInput;
                
                // BACKPROP: calculate error
                for(int level=nbLayer-1; level>=0; level--) {
                    for(int neuron=0; neuron<network[level].length; neuron++) {
                        if(level==nbLayer-1) {
                            if(attrClass.isNumeric()) {
                                network[level][neuron].setError(targets[neuron]-output[neuron]);
                            }
                            else {
                                network[level][neuron].setError((targets[neuron]-output[neuron])*(1-output[neuron])*output[neuron]);
                            }
                        }
                        else {
                            double error = output[neuron]*(1-output[neuron]);
                            double sumWeightError = 0.0;
                            for(int j=0; j<network[level+1].length; j++) {
                                sumWeightError += network[level+1][j].getError()*network[level+1][j].getSpecificWeight(neuron);
                            }
                            network[level][neuron].setError(error*sumWeightError);
                        }
                    }
                }
                
                // UPDATE WEIGHT
                if(in==0) {
                    for(int level=0; level<network.length; level++) {
                        for(int neuron=0; neuron<network[level].length; neuron++) {
                            network[level][neuron].updateWeight(learningRate);
                        }
                    }
                }
                else {
                    for(int level=0; level<network.length; level++) {
                        for(int neuron=0; neuron<network[level].length; neuron++) {
                            network[level][neuron].updateWeightWithPrevious(learningRate,momentum);
                        }
                    }
                }
            }
            // RECALCULATE OUTPUT PER NEURON
            for (int in=0; in<data.numInstances(); in++) {
                Instance instance = data.instance(in);
                double[] output;                        // output masing2 class layer
                double[] localInput = new double[nbInput];
                for(int ins=0; ins<localInput.length; ins++) {
                    localInput[ins] = instance.value(ins);
                }
                for(int level=0; level<network.length; level++) {
                    double[] result = new double[network[level].length];
                    for(int neuron=0; neuron<network[level].length; neuron++) {
                        network[level][neuron].setInput(localInput);
                        network[level][neuron].computeValue();
                        result[neuron] = network[level][neuron].getValue();
                    }
                    localInput = result;
                }
                output = localInput;
                double mse = 0.0;
                for (int iter=0; iter<output.length; iter++) {
                    mse += Math.pow(targets[iter]-output[iter], 2);
                }
                mse /= 2.0;
                MSE += mse;
            }
            MSE /= ((data.numInstances()-1)*network[network.length-1].length);
            epoch++;
        }
    }
    
}
