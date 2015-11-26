/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author LINDA
 */
public class Node {
    private float nodeVal;          //nilai pada node
    private List nodeIn = new ArrayList();     //list node yang menjadi input
    private List nodeOut = new ArrayList();    //list node yang menjadi output
    private  float weight;          //nilai weight node terhadap node output
    
    public Node(){
        
    }
    
    /*Getter*/
    public float getNodeVal(){      //mengembalikan nilai nodeVal
        return nodeVal;         
    }
    public Object getNodeIn(int i){
        Object object = nodeIn.get(i);
        return object;
    }
    public float getWeight(int i){
        Object object = nodeIn.get(i);
        Node node = (Node) object;
        return node.weight;
    }
    
}
