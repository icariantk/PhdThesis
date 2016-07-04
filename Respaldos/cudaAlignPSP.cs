   public Sequencer cudaAlignPSP(Sequencer a, Sequencer b, String submatrix = "blosum62")
        {
            string result = "";
            string[] aligned;
            int am = a.sequences.Count;
            int an = a.longest() + 1;
            int bm = b.sequences.Count;
            int bn = b.longest() + 1;
            //Console.WriteLine("an:" + an + "  bn:" + bn);
            int[] SeqA = new int[am * an];
            int[] SeqB = new int[bm * bn];
            int[] inv_Seqs = new int[((am + bm) * (an + bn))];
            int[] alignedSeqs = new int[((am + bm) * (an + bn))];
            int[] matrix = new int[an * bn];
            int[] matrixDir = new int[an * bn];
            int[] scoreMatrix = new int[729];
            int[] size = new int[1] { 0 };

            //Transform 2D Scorematrix to vector
            for (int x = 0; x != 27; x++)
            {
                for (int y = 0; y != 27; y++)
                {
                    scoreMatrix[y * 27 + x] = Blosum62[x][y];
                }
            }

            using (CudaContext cntxt = new CudaContext())
            {
/*
                CUmodule CUmodule = cntxt.LoadModulePTX("kernel.ptx");
                CudaKernel alignPSP = new CudaKernel(getKernelName(@".\kernel.ptx", "alignPSP"), CUmodule, cntxt) { GridDimensions = new dim3(1, 1), BlockDimensions = new dim3(256, 1) };
                CudaKernel tracebackPSP = new CudaKernel(getKernelName(@".\kernel.ptx", "tracebackPSP"), CUmodule, cntxt) { GridDimensions = new dim3(1, 1), BlockDimensions = new dim3(1, 1) };
                CudaKernel invertPSP = new CudaKernel(getKernelName(@".\kernel.ptx", "invertPSP"), CUmodule, cntxt) { GridDimensions = new dim3(1, 1), BlockDimensions = new dim3(256, 1) };
                
              */
                byte[] buff = null;
                FileStream fs = new FileStream(@".\kernel.ptx",
                                               FileMode.Open,
                                               FileAccess.Read);
                BinaryReader br = new BinaryReader(fs);
                long numBytes = new FileInfo(@".\kernel.ptx").Length;
                buff = br.ReadBytes((int)numBytes);


                CudaKernel alignPSP = cntxt.LoadKernelPTX(buff, getKernelName(@".\kernel.ptx", "alignPSP"));
                alignPSP.BlockDimensions = new dim3(256);
                alignPSP.GridDimensions = new dim3(1);
                alignPSP.DynamicSharedMemory = Convert.ToUInt32((an + bn) * 128);
                

                CudaKernel tracebackPSP = cntxt.LoadKernelPTX("kernel.ptx", getKernelName(@".\kernel.ptx", "tracebackPSP"));
                CudaKernel invertPSP = cntxt.LoadKernelPTX("kernel.ptx", getKernelName(@".\kernel.ptx", "invertPSP"));

                Console.WriteLine(alignPSP.KernelName);
                Console.WriteLine(tracebackPSP.KernelName);
                Console.WriteLine(invertPSP.KernelName);

                CudaDeviceVariable<int> SeqA_d = new CudaDeviceVariable<int>(am * an);
                CudaDeviceVariable<int> sizeAligned = new CudaDeviceVariable<int>(1);
                CudaDeviceVariable<int> SeqB_d = new CudaDeviceVariable<int>(bm * bn);

                CudaDeviceVariable<int> matrix_d = new CudaDeviceVariable<int>(an * bn);
                CudaDeviceVariable<int> matrixDir_d = new CudaDeviceVariable<int>(an * bn);
                CudaDeviceVariable<int> scoreMatrix_d = new CudaDeviceVariable<int>(729);

                CudaDeviceVariable<int> SeqInv_d = new CudaDeviceVariable<int>((am + bm) * (an + bn));
                CudaDeviceVariable<int> alignedSeqs_d = new CudaDeviceVariable<int>((am + bm) * (an + bn));
                /*for (int x = 0; x != ((am + bm) * (an + bn)); x++)
                {
                    alignedSeqs[x] = '$';
                    inv_Seqs[x] = '$';
                }
                alignedSeqs_d.CopyToDevice(alignedSeqs);
                SeqInv_d.CopyToDevice(inv_Seqs);
                */

                

                //scoreMatrix_d.CopyToDevice(scoreMatrix);
                scoreMatrix_d = scoreMatrix;
                for (int y = 0; y != am; y++)
                {
                    for (int x = 0; x != an - 1; x++)
                    {
                        SeqA[y * an + x] = a.sequences.ElementAt(y).ElementAt(x);
                    }
                }
                for (int y = 0; y != bm; y++)
                {
                    for (int x = 0; x != bn - 1; x++)
                    {
                        SeqB[y * bn + x] = b.sequences.ElementAt(y).ElementAt(x);
                    }
                }

                
                SeqA_d = SeqA;
                SeqB_d = SeqB;
                
                
                 //alignPSP.Run(SeqA_d.DevicePointer, SeqB_d.DevicePointer, matrix_d.DevicePointer, matrixDir_d.DevicePointer, scoreMatrix_d.DevicePointer, am, an, bm, bn, gap(0), gap(1), gap(2), 3,256);
                alignPSP.Run();
                    
                  
                
                
                for (int n = 0; n != (int)(SeqA.Length / 256)+1; n++)
                    tracebackPSP.Run(SeqInv_d.DevicePointer, SeqA_d.DevicePointer, SeqB_d.DevicePointer, matrixDir_d.DevicePointer, am, an, bm, bn, sizeAligned.DevicePointer,n);

                sizeAligned.CopyToHost(size);

                /* matrixDir_d.CopyToHost(matrixDir);
                 matrix_d.CopyToHost(matrix);*/
                /*
                Console.Write("Matrix: \n");
                for (int c = 0; c != bn; c++)
                {
                    for (int cc = 0; cc != an; cc++)
                    {
                        Console.Write(matrix[(c * an) + cc] + "\t");
                    }
                    Console.Write("\n");
                }*/

                /*Console.Write("MatrixDir: \n");
                for (int c = 0; c != bn; c++)
                {
                    for (int cc = 0; cc != an; cc++)
                    {
                        Console.Write(matrixDir[(c * an) + cc] + " ");
                    }
                    Console.Write("\n");
                }
                */
                
                invertPSP.Run(alignedSeqs_d.DevicePointer, SeqInv_d.DevicePointer, (am + bm), sizeAligned.DevicePointer, (an + bn));

                //                matrixDir_d.CopyToHost(matrixDir);
                /*

                Console.WriteLine("an: " + an + "am: " + am + "bn: " + bn + "bm: " + bm);
                 */

                alignedSeqs_d.CopyToHost(alignedSeqs);

                for (int c = 0; c != am + bm; c++)
                {
                    if (c < am)
                    {
                        result += a.getHeader(c) + "\n";
                    }
                    else
                    {
                        result += b.getHeader(c - am) + "\n";
                    }

                    for (int cc = 0; cc != size[0]; cc++)
                    {
                        result += Convert.ToChar(alignedSeqs[(an + bn) * c + cc]);
                    }
                    result += "\n";
                }

                /*timer.Stop();
                Console.WriteLine("Size aligned: " + size[0]);
                Console.WriteLine("In function: " + timer.ElapsedTicks);*/
            }

            aligned = result.Split('|');

            Sequencer res = new Sequencer(result, false);

            return res;
        }
