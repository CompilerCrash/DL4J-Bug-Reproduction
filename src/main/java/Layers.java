import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Layers {

    private static final int b = 2;
    private static final int n = 4;

    public static void main(String[] args) {
        String separator = "=".repeat(60);

        nd4jLinearLayerBiasNull();
        System.out.println(separator);
        sdLinearLayerBiasNull();
        System.out.println(separator);
        nd4jReluLayerBiasNull();
        System.out.println(separator);
        sdReluLayerBiasNull();
        System.out.println(separator);
        nd4jReluLayerBatch1();
        System.out.println(separator);
        nd4jReluLayerNIn1();
        System.out.println(separator);
        nd4JReluLayerNOut1();
        System.out.println(separator);
        sdReluLayerBatch1();
        System.out.println(separator);
        sdReluLayerNIn1();
        System.out.println(separator);
        sdReluLayerNOut1();
    }

    public static void nd4jLinearLayerBiasNull() {
        INDArray res = Nd4j.nn.linear(Nd4j.ones(b, n), Nd4j.ones(n, n), null); // throws exception
        System.out.println(res);
    }

    public static void sdLinearLayerBiasNull() {
        SameDiff sd = SameDiff.create();
        SDVariable res = sd.nn.linear(sd.one(null, b, n), sd.one(null, n, n), null); // throws exception
        System.out.println(res.eval());
    }

    public static void nd4jReluLayerBiasNull() {
        INDArray res = Nd4j.nn.reluLayer(Nd4j.ones(b, n), Nd4j.ones(n, n), null); // throws exception
        System.out.println(res);
    }

    public static void sdReluLayerBiasNull() {
        SameDiff sd = SameDiff.create();
        SDVariable res = sd.nn.reluLayer(sd.one(null, b, n), sd.one(null, n, n), null); // throws exception
        System.out.println(res.eval());
    }

    public static void nd4jReluLayerBatch1() {
        INDArray res = Nd4j.nn.reluLayer(Nd4j.ones(1, n), Nd4j.ones(n, n), Nd4j.zeros(n)); // throws exception
        System.out.println(res);
    }

    public static void nd4jReluLayerNIn1() {
        INDArray res = Nd4j.nn.reluLayer(Nd4j.ones(b, 1), Nd4j.ones(1, n), Nd4j.zeros(n)); // throws exception
        System.out.println(res);
    }

    public static void nd4JReluLayerNOut1() {
        INDArray res = Nd4j.nn.reluLayer(Nd4j.ones(b, n), Nd4j.ones(n, 1), Nd4j.zeros(1)); // throws exception
        System.out.println(res);
    }

    public static void sdReluLayerBatch1() {
        SameDiff sd = SameDiff.create();
        SDVariable res = sd.nn.reluLayer(sd.one(null, 1, n), sd.one(null, n, n), sd.zero(null, n)); // throws exception
        System.out.println(res.eval());
    }

    public static void sdReluLayerNIn1() {
        SameDiff sd = SameDiff.create();
        SDVariable res = sd.nn.reluLayer(sd.one(null, b, 1), sd.one(null, 1, n), sd.zero(null, n)); // throws exception
        System.out.println(res.eval());
    }

    public static void sdReluLayerNOut1() {
        SameDiff sd = SameDiff.create();
        SDVariable res = sd.nn.reluLayer(sd.one(null, b, n), sd.one(null, n, 1), sd.zero(null, 1)); // throws exception
        System.out.println(res.eval());
    }
}
