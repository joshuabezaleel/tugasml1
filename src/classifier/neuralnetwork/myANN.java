/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author Rakhmatullah Yoga S, Linda Sekawati, Joshua Bezaleel Abednego
 */
public class myANN extends Classifier {
    private enum Function {
        SIGN, STEP, SIGMOID 
    }
    private Function func;
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
    private int epoch;
    private boolean singlePerceptron;
    private boolean randomWeight;
    
    /**
     * mendapatkan sum hasil kali weight dan x dari input layer sebelumnya
     * @param idxLayer index dari layer yang akan menerima input
     * @param layerLevel level dari layer sebelumnya yang menjadi input
     * @return 
     */
    public double countInput(int idxLayer, int layerLevel) {
        double outputSum = 0;
        for (Node network_layerN : network[layerLevel]) {
            outputSum += network_layerN.getOutput(idxLayer);
        }
        return activationFunction(outputSum);
    }
    
    /**
     * Fungsi aktivasi untuk masing-masing neuron
     * @param sum jumlah hasil kali bobot dan value
     * @return hasil masukan fungsi aktivasi
     */
    public double activationFunction(double sum) {
        double result;
        switch(func) {
            case SIGN:
                if(sum>=0)
                    result = 1;
                else
                    result = -1;
                break;
            case STEP:
                if(sum>=threshold)
                    result = 1;
                else
                    result = 0;
                break;
            case SIGMOID:
                result = 1/(1+Math.exp(-sum));
                break;
        }
        return 0;
    }
    
    public double countError(int idxLevel) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        NominalToBinary nomToBinFilter = new NominalToBinary();
        Normalize normalizeFilter = new Normalize();
        
        data = new Instances(data);
        data.deleteWithMissingClass();
        nomToBinFilter.setInputFormat(data);
        data = Filter.useFilter(data, nomToBinFilter);
        normalizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, normalizeFilter);
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
                network[i][j].initWeight(nbNeuron[i+1], givenWeight);
                if(randomWeight)
                    network[i][j].randomizeWeight();
            }
        }
        attrClass = data.classAttribute();
    }
    
}
