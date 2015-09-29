/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree_exploration.myC45Utils;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Rakhmatullah Yoga S
 */
public abstract class myC45ClassifierSplitModel {
    protected myDistribution dist;
    protected int nbSubset;
    
    public abstract void buildClassifier(Instances instances) throws Exception;
    public abstract double [] getWeights(Instance instance);
    public abstract int getSubset(Instance instance) throws Exception;
    public final int nbSubset() {
        return nbSubset;
    }
    public final Instances [] split(Instances data) throws Exception {
        Instances [] instances = new Instances [nbSubset];
        double [] weights;
        double newWeight;
        Instance instance;
        int subset;
        
        for (int i = 0; i < nbSubset; i++) {
            instances[i] = new Instances(data, data.numInstances());
        }
        for(int j = 0; j < data.numInstances(); j++) {
            instance = data.instance(j);
            weights = getWeights(instance);
            subset = getSubset(instance);
            if(subset > -1) {
                instances[subset].add(instance);
            }
            else {
                for(int i = 0; i < nbSubset; i++) {
                    if(Utils.gr(weights[i], 0)) {
                        newWeight = weights[i]*instance.weight();
                        instances[i].add(instance);
                        instances[i].lastInstance().setWeight(newWeight);
                    }
                }
            }
        }
        for (int i = 0; i < nbSubset; i++) {
            instances[i].compactify();
        }
        return instances;
    }
}
