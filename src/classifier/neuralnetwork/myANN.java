/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Rakhmatullah Yoga S
 */
public class myANN extends Classifier {
    private float deltaW;                           //nilai delta dari weight
    private List listInput = new ArrayList();       //list fitur-fitur pada satu input
    private List listNodeLayer  = new ArrayList();  //list node pada sebuah layer
    private List listInstance = new ArrayList();    //list instance
    private List targetVal  = new ArrayList();      //list nilai target
    private int classVal;
    public myANN() {
        
    }

    
    public float getDeltaW(){                   //mengembalikan nilai delta dari weight yang akan di-update
        return deltaW;
    }
        
    public void setDeltaWeight(float deltaW){   //set nilai delta dari weight
        
        this.deltaW = deltaW;
    }
    
    public void errorOutput(){
        
    }
    
    public void makeNode(ArrayList listInput){
        for(int i=0; i<this.classVal; i++){
            Node node = new Node();
            Iterator iterator = listInput.iterator();
            while(iterator.hasNext()){
                Object object = iterator.next();
                Node nodeInput = (Node) object;
                nodeInput.setWeight(0);
            }
            this.listNodeLayer.add(node);
        }
    }
    
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public double classifyInstance(Instance instance) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
