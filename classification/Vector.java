import java.text.DecimalFormat;
 
/* slights modification added to http://introcs.cs.princeton.edu/java/33design/Vector.java.html */


public class Vector { 

    private final int N;         // length of the vector
    private double[] data;       // array of vector's components

    // create the zero vector of length N
    public Vector(int N) {
        this.N = N;
        this.data = new double[N];
    }
    
    // create a copy
    public Vector(Vector v) {
        this.N = v.N;
        this.data = v.data;
    }
    
    // create the one vector of length N
    public Vector(int N, int one) {
        this.N = N;
        this.data = new double[N];
        for (int i = 0; i < N; i++)
            this.data[i] = 1.0;
    }

    // create a vector from an array
    public Vector(double[] data) {
        N = data.length;

        // defensive copy so that client can't alter our copy of data[]
        this.data = new double[N];
        for (int i = 0; i < N; i++)
            this.data[i] = data[i];
    }
    
    // create a vector from a String separated by a tabulation
    public Vector(String s) {
    	String[] weights = s.split("\t");
  	  	double[] wArray = new double[weights.length];
  	  	for(int i=0; i<weights.length; i++){
  	  		wArray[i] = Double.parseDouble(weights[i]);
  	  	}
  	  	this.N = weights.length;
  	  	this.data = wArray;
    }

    // return the length of the vector
    public int length() {
        return N;
    }

    // return the inner product of this Vector a and b
    public double dot(Vector that) {
        if (this.N != that.N) throw new RuntimeException("Dimensions don't agree");
        double sum = 0.0;
        for (int i = 0; i < N; i++)
            sum = sum + (this.data[i] * that.data[i]);
        return sum;
    }
    
    // return sum_1:p [ abs(x_i*y_i) ]
    public double dot2(Vector that) {
        if (this.N != that.N) throw new RuntimeException("Dimensions don't agree");
        double sum = 0.0;
        for (int i = 0; i < N; i++)
            sum = sum + Math.abs(this.data[i] * that.data[i]);
        return sum;
    }

    // return the Euclidean norm of this Vector
    public double magnitude() {
        return Math.sqrt(this.dot(this));
    }
    
    // return the L1 norm of this Vector
    public double magnitudeL1() {	
        return this.dot2(new Vector(this.N, 1));
    }

    // return the Euclidean distance between this and that
    public double distanceTo(Vector that) {
        if (this.N != that.N) throw new RuntimeException("Dimensions don't agree");
        return this.minus(that).magnitude();
    }
    
    // return the L1 distance between this and that
    public double distanceToL1(Vector that) {
        if (this.N != that.N) throw new RuntimeException("Dimensions don't agree");
        return this.minus(that).magnitudeL1();
    }
    
    // return this + that
    public Vector plus(Vector that) {
        if (this.N != that.N) throw new RuntimeException("Dimensions don't agree");
        Vector c = new Vector(N);
        for (int i = 0; i < N; i++)
            c.data[i] = this.data[i] + that.data[i];
        return c;
    }

    // return this - that
    public Vector minus(Vector that) {
        if (this.N != that.N) throw new RuntimeException("Dimensions don't agree");
        Vector c = new Vector(N);
        for (int i = 0; i < N; i++)
            c.data[i] = this.data[i] - that.data[i];
        return c;
    }

    // return the corresponding coordinate
    public double cartesian(int i) {
        return data[i];
    }

    // create and return a new object whose value is (this * factor)
    public Vector times(double factor) {
        Vector c = new Vector(N);
        for (int i = 0; i < N; i++)
            c.data[i] = factor * data[i];
        return c;
    }

    // return the corresponding unit vector
    public Vector direction() {
        if (this.magnitude() == 0.0) throw new RuntimeException("Zero-vector has no direction");
        return this.times(1.0 / this.magnitude());
    }
    
    // return a string representation of the vector
    public String toString() {
        String s = "";
        for (int i = 0; i < N; i++) {
            s += data[i];
            if (i < N-1) s+= "\t"; 
        }
        return s;
    }
    
    // return a string representation of the vector with a format
    public String toPrettyString() {
        String s = "";
        DecimalFormat df = new DecimalFormat("###.##");
        for (int i = 0; i < N; i++) {
            s += df.format(data[i]);
            if (i < N-1) s+= "\t"; 
        }
        return s;
    }

}
