import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class ExpandDims {

    public static void main(String[] args) {
        String separator = "=".repeat(40);

        expand2dTo3d();
        System.out.println(separator);
        expandLastAxisOfScalar();
        System.out.println(separator);
        expandLastAxis();
        System.out.println(separator);
        nd4jExpandOutOfBounds();
        System.out.println(separator);
        sdExpandOutOfBounds();
    }

    public static void expand2dTo3d() {
        SameDiff sd = SameDiff.create();
        SDVariable v1 = sd.zero(null, 1, 1);
        SDVariable v2 = sd.expandDims(v1, 2);
        System.out.println(v1.shape().eval()); // [1, 1]
        System.out.println(v2.shape().eval()); // should be [1, 1, 1] but is [1, 1]
    }

    public static void expandLastAxisOfScalar() {
        INDArray v1 = Nd4j.scalar(1);
        System.out.println(Arrays.toString(v1.shape())); // []
        INDArray v2 = Nd4j.expandDims(v1, -1); // throws exception
        System.out.println(Arrays.toString(v2.shape())); // should now be [1]
    }

    // FIXME
    public static void expandLastAxis() {
        INDArray v1 = Nd4j.ones(2, 2);
        System.out.println(Arrays.toString(v1.shape())); // [2, 2]
        INDArray v2 = Nd4j.expandDims(v1, 2); // throws exception
        System.out.println(Arrays.toString(v2.shape())); // should now be [2, 2, 1]
    }

    public static void nd4jExpandOutOfBounds() {
        try {
            INDArray v1 = Nd4j.ones(2, 2);
            INDArray v2 = Nd4j.expandDims(v1, 3); // crashes
        } catch (ND4JIllegalStateException e) {
            e.printStackTrace();
        }
    }

    public static void sdExpandOutOfBounds() {
        try {
            SameDiff sd = SameDiff.create();
            SDVariable v1 = sd.one(null, 2, 2);
            SDVariable v2 = sd.expandDims(v1, 3);
            v2.shape().eval(); // crashes
        } catch (ND4JIllegalStateException e) {
            e.printStackTrace();
        }
    }
}
