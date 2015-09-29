/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree_exploration;

import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 *
 * @author Rakhmatullah Yoga S
 */
public class myC45 extends Classifier {
    
    private myC45[] successor;
    
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
}
