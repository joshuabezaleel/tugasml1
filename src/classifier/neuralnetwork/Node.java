/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

/**
 *
 * @author LINDA
 */
public class Node {
    private float nodeVal;          //nilai pada node
    // List private int nodeIn;     //list node yang menjadi input
    // List private int nodeOut;    //list node yang menjadi output
    private  float weight;          //nilai weight node terhadap node output
    
    public Node(){
        
    }
    
    /*Getter*/
    public float getNodeVal(){      //mengembalikan nilai nodeVal
        return nodeVal;         
    }
    
    
}
