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
    
    public double countError(int idxLevel) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
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
        attrClass = data.classAttribute();
        int epoch=0;
        double MSE = 0;
        while(epoch<maxEpoch && MSE>threshold) {
            for (int in=0; in<data.numInstances(); in++) {
                Instance instance = data.instance(in);
                double[] targets;
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
//                double currentMSE = 0.0;
//                for(int id=0; id<targets.length; id++) {
//                    currentMSE+=Maths.square(targets[id]-output[id]);
//                }
//                currentMSE/=2.0;
                // BACKPROP
                
            }
            
            epoch++;
        }
    }
    
}
