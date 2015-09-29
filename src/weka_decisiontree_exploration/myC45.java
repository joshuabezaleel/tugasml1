/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree_exploration;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka_decisiontree_exploration.myC45Utils.myC45ClassifierSplitModel;

/**
 *
 * @author Rakhmatullah Yoga S
 */
public class myC45 extends Classifier {
    
    private myC45[] successor;
    
    private myC45ClassifierSplitModel local_model;
    
    private boolean is_leaf;
    
    private boolean use_pruning = false;
    
    public void setPruning(boolean prune) {
        use_pruning = prune;
    }
    
    @Override
    public void buildClassifier(Instances i) throws Exception {
        Instances data = new Instances(i);
        data.deleteWithMissingClass();
        //buildTree
    }
    
    public void buildTree() {
        
    }
    
    public double classifyInstance() {
        return 0;
    }
    
    public double getProbabilities(int idxClass, Instance instance, double weight) {
        double prob = 0;
        if(is_leaf) {
            return weight * getLocalModel()
        }
    }
    
    public myC45ClassifierSplitModel getLocalModel() {
        return (myC45ClassifierSplitModel) local_model;
    }
}
