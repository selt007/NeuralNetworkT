using System;
using System.Threading;

namespace NNT
{
    class Program
    {
        static void Main(string[] args)
        {
            Formula meth;
            meth = new Formula();

            Console.Clear();

            //double w1 = 0.45, w2 = 0.78,
            //    w3 = -0.12, w4 = 0.13,
            //    w5 = 1.5, w6 = -2.3,
            //    result = 0, error = 0, ideal = 1,
            //    Do, Dh1, Dh2, GRADw5, GRADw6;

            //const double E = 0.7, A = 0.3;

            //double[] Dw5;
            //double[] Dw6;

            //int I1 = 1, I2 = 0;

            //double h1 = Math.Round(
            //    meth.Sigmoid(meth.InOut(
            //        w1, w3, I1, I2)), 2);

            //double h2 = Math.Round(
            //    meth.Sigmoid(meth.InOut(
            //        w2, w4, I1, I2)), 2);

            double result = 0, error = 0, ideal = 1;
            int maxEpoch = 1, trainSet = 20;
            double[] w13 = new double[] { 0, 0, 0.45, -0.12 };
            double[] w24 = new double[] { 0, 0, 0.78, 0.13 };
            double[] w56 = new double[] { 0, 0, 1.5, -2.3 };
            double[] Dw1 = new double[] { 0, 0 };
            double[] Dw2 = new double[] { 0, 0 };
            double[] Dw3 = new double[] { 0, 0 };
            double[] Dw4 = new double[] { 0, 0 };
            double[] Dw5 = new double[] { 0, 0 };
            double[] Dw6 = new double[] { 0, 0 };
            const double E = 0.7, A = 0.3;
            int[] I12 = new int[] { 1, 0 };
            double Do, Dh1, Dh2, 
                GRADw5, GRADw6,
                GRADw1, GRADw2,
                GRADw3, GRADw4;

            for (int i = 0; i < maxEpoch;i++)
            {
                double h1 = Math.Round(
                    meth.Sigmoid(meth.InOut(
                        w13, I12)), 2);

                double h2 = Math.Round(
                    meth.Sigmoid(meth.InOut(
                        w24, I12)), 2);

                double[] arr = new double[] { 0, 0, h1, h2 };

                result = Math.Round(
                    meth.Sigmoid(
                        meth.HideOut(arr, w56)),
                    2);
                error = Math.Round(
                    Math.Pow(
                        (1 - result), 2), 2);

                for (int j = 0; j < trainSet;j++)
                {
                    Do = (1 - error) * ((ideal - error) * error);

                    Dh1 = ((1 - h1) * h1) * (w56[2] * Do);
                    GRADw5 = h1 * Do;
                    Dw5[1] = E * GRADw5 + Dw5[0] * A;
                    w56[2] += Dw5[1];

                    Dh2 = ((1 - h2) * h2) * (w56[3] * Do);
                    GRADw6 = h2 * Do;
                    Dw6[1] = E * GRADw6 + Dw6[0] * A;
                    w56[3] += Dw6[1];

                    GRADw1 = I12[0] * Dh1;
                    GRADw2 = I12[0] * Dh2;
                    GRADw3 = I12[1] * Dh1;
                    GRADw4 = I12[1] * Dh2;

                    Dw1[1] = E * GRADw1 + Dw1[0] * A;
                    Dw2[1] = E * GRADw2 + Dw2[0] * A;
                    Dw3[1] = E * GRADw3 + Dw3[0] * A;
                    Dw4[1] = E * GRADw4 + Dw4[0] * A;

                    w13[2] += Dw1[1];
                    w13[3] += Dw2[1];
                    w24[2] += Dw3[1];
                    w24[3] += Dw4[1];
                }

                Console.WriteLine("Step " + i);
                Console.WriteLine("-----------------");
                Console.WriteLine("Result - " + result + " , |");
                Console.WriteLine("Error - " + error + " .  |");
                Console.WriteLine("----------------- ");

                Thread.Sleep(250);

                if (error == 0) break;
                else { maxEpoch++; Console.Clear(); }
            }
            
            Console.ReadKey();
        }
    }
}
