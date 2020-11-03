#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
Function to calculate the 1-Norm of a qubit Hamiltonian after Jordan-Wigner,
without doing the expensive Jordan-Wigner transformation.
"""
import numpy as np

def normal_order_tbc(two_body_coefficients, n_qubits):
    '''
    Normal order the Hamiltonian giving the right two-body coefficients

    Parameters
    ----------
    two_body_coefficients : A numpy array of size
        (n_qubits, n_qubits, n_qubits, n_qubits)
    n_qubits : Number of qubits

    Returns
    -------
    two_body_coefficients : A numpy array of size
        (n_qubits, n_qubits, n_qubits, n_qubits)
    '''
    # print("Normal ordering.....")
    for i in range(n_qubits):
        for j in range(n_qubits):
            for k in range(n_qubits):
                for l in range(n_qubits):
                    if i == j or k == l:
                        two_body_coefficients[i,j,k,l] = 0.
                    elif i > j and k > l:
                        two_body_coefficients[i,j,k,l] = (
                        two_body_coefficients[i,j,k,l] -
                        two_body_coefficients[i,j,l,k] -
                        two_body_coefficients[j,i,k,l] +
                        two_body_coefficients[j,i,l,k])
                        
                        two_body_coefficients[i,j,l,k] = 0.
                        two_body_coefficients[j,i,k,l] = 0.
                        two_body_coefficients[j,i,l,k] = 0.
    # print("Done normal ordering")
    return two_body_coefficients

def JW1norm_nosym(constant, one_body_coefficients, two_body_coefficients, normal_order=True):
    '''
    Returns the 1-Norm of the Hamiltonian after a Jordan-Wigner
    transformation given normal ordered one-body (2D np.array)
    and two-body (4D np.array) coefficients.

    Parameters
    ----------
    constant : Nuclear repulsion or adjustment to constant shift in Hamiltonian
            from integrating out core orbitals
    one_body_coefficients : An array of the one-electron integrals having
                shape of (n_qubits, n_qubits).
    two_body_coefficients : An array of the two-electron integrals having
                shape of (n_qubits, n_qubits, n_qubits, n_qubits).
    normal_order : Boolean, optional
        Whether to normal order the Hamiltonian (If false, assumes that
        the Hamiltonian is already in normal ordered form). The default is True.

    Returns
    -------
    q1norm : 1-Norm of the Qubit Hamiltonian  
    '''
    n_qubits = one_body_coefficients.shape[0]
    
    if normal_order:
        two_body_coefficients = normal_order_tbc(two_body_coefficients, n_qubits)
        
    q1norm = 0           

    # p = r, q = s
    cs = [0 for i in range((n_qubits+2))]
    cs[-1] = -4*constant
    for i in range(n_qubits):
        cs[-1] -= 2* one_body_coefficients[i,i]
        for j in range(i):                                         
            cs[0] += abs(two_body_coefficients[i,j,i,j])
            cs[-1] += two_body_coefficients[i,j,i,j]
    for i in range(n_qubits):
        cs[i+1] -= 2 * one_body_coefficients[i,i]
        for j in range(n_qubits):
            if j > i:
                cs[i+1] += two_body_coefficients[j,i,j,i]
                
            elif j < i:
                cs[i+1] += two_body_coefficients[i,j,i,j]
    q1norm += 1/4 * sum(list(map(abs,cs)))
    
    # p != r != q != s
    for i in range(n_qubits):
        for j in range(i):
            for k in range(j):
                for l in range(k):
                    ijkl = two_body_coefficients[i,j,k,l]
                    ikjl = two_body_coefficients[i,k,j,l]
                    iljk = two_body_coefficients[i,l,j,k]
                    jkil = two_body_coefficients[j,k,i,l]
                    jlik = two_body_coefficients[j,l,i,k]
                    klij = two_body_coefficients[k,l,i,j]
                    q1norm += 1/4 * max(abs(ijkl + ikjl + iljk),
                                        abs(jkil + jlik + klij))
                    q1norm += 1/4 * max(abs(-ijkl + ikjl + iljk),
                                        abs(jkil + jlik - klij))
                    q1norm += 1/4 * max(abs(ijkl - ikjl + iljk),
                                        abs(jkil - jlik + klij))
                    q1norm += 1/4 * max(abs(ijkl + ikjl - iljk),
                                        abs(-jkil + jlik + klij))
    
    # p = r or q = s
    for i in range(n_qubits):
        for j in range(i):
            temp_a = - 2 * one_body_coefficients[i,j]
            temp_b = - 2 * one_body_coefficients[j,i]
            for k in range(j):
                temp_a += two_body_coefficients[i,k,j,k]
                temp_b += two_body_coefficients[j,k,i,k]
                q1norm += 1/2 * max(abs(two_body_coefficients[i,k,j,k]),\
                                    abs(two_body_coefficients[j,k,i,k]))
                
            for k in range(j+1,i):
                temp_a -= two_body_coefficients[i,k,k,j]
                temp_b -= two_body_coefficients[k,j,i,k]
                q1norm += 1/2 * max(abs(two_body_coefficients[i,k,k,j]),\
                                    abs(two_body_coefficients[k,j,i,k]))

            for k in range(i+1,n_qubits):
                temp_a += two_body_coefficients[k,i,k,j]
                temp_b += two_body_coefficients[k,j,k,i]
                q1norm += 1/2 * max(abs(two_body_coefficients[k,i,k,j]),\
                                    abs(two_body_coefficients[k,j,k,i]))

            q1norm += 1/2 * max(abs(temp_a), abs(temp_b))

    return q1norm

def JW1norm(constant, one_body_coefficients, two_body_coefficients_inp, normal_order=False):
    '''
    Returns the 1-Norm of the Hamiltonian after a Jordan-Wigner
    transformation given normal ordered one-body (2D np.array)
    and two-body (4D np.array) coefficients.

    Parameters
    ----------
    constant : Nuclear repulsion or adjustment to constant shift in Hamiltonian
            from integrating out core orbitals
    one_body_coefficients : An array of the one-electron integrals having
                shape of (n_qubits, n_qubits).
    two_body_coefficients : An array of the two-electron integrals having
                shape of (n_qubits, n_qubits, n_qubits, n_qubits).
    normal_order : Boolean, optional
        Whether to normal order the Hamiltonian (If false, assumes that
        the Hamiltonian is already in normal ordered form). The default is True.

    Returns
    -------
    q1norm : 1-Norm of the Qubit Hamiltonian  
    '''
    n_qubits = one_body_coefficients.shape[0]
    if normal_order:
        two_body_coefficients = normal_order_tbc(two_body_coefficients_inp, n_qubits)
    else:
        two_body_coefficients = 2*np.copy(two_body_coefficients_inp)
    

    
    htilde = constant
    for p in range(n_qubits):
        htilde += 1/2 * one_body_coefficients[p,p]
        for q in range(n_qubits):
            if q != p:
                htilde += 1/8 * (two_body_coefficients[p,q,q,p] - two_body_coefficients[p,q,p,q])
    
    htildepq = np.zeros(one_body_coefficients.shape)
    for p in range(n_qubits):
        for q in range(n_qubits):
            htildepq[p,q] = 1/2 * one_body_coefficients[p,q]
            for r in range(n_qubits):
                if r != p and r!= q:
                    htildepq[p,q] += ((1/4 * two_body_coefficients[p,r,r,q]) - \
                                      (1/4 * two_body_coefficients[p,r,q,r]))
    
    q1norm1 = abs(htilde) + np.sum(np.absolute(np.diag(htildepq)))
    q1norm3 = 0
    for p in range(n_qubits):
        for q in range(n_qubits):
            if p != q:
                q1norm3 += abs(htildepq[p,q])
    q1norm2 = 0
    for p in range(n_qubits):
        for q in range(n_qubits):
            if p != q:
                q1norm1 += 1/8 * abs(two_body_coefficients[p,q,p,q]-two_body_coefficients[p,q,q,p])
            for r in range(n_qubits):
                if p != q and q!= r and p!=r:
                    q1norm3 += 1/4 * abs(two_body_coefficients[p,r,q,r]-two_body_coefficients[p,r,r,q])
                for s in range(n_qubits):
                    if p>q and r>s and p!=q and p!=r and p!=s and q!=r and q!=s and r!=s:
                        q1norm2 += 1/4 * abs(two_body_coefficients[p,q,r,s] - \
                                                two_body_coefficients[p,q,s,r])
    q1norm = q1norm1 + q1norm2 + q1norm3
    return q1norm

def JW1norm_spat(constant, one_body_integrals, two_body_integrals_inp, normal_order=False):
    '''
    Returns the 1-Norm of the Hamiltonian after a Jordan-Wigner
    transformation given normal ordered one-body (2D np.array)
    and two-body (4D np.array) integrals.

    Parameters
    ----------
    constant : Nuclear repulsion or adjustment to constant shift in Hamiltonian
            from integrating out core orbitals
    one_body_integrals : An array of the one-electron integrals having
                shape of (n_qubits, n_qubits).
    two_body_integrals : An array of the two-electron integrals having
                shape of (n_qubits, n_qubits, n_qubits, n_qubits).
    normal_order : Boolean, optional
        Whether to normal order the Hamiltonian (If false, assumes that
        the Hamiltonian is already in normal ordered form). The default is True.

    Returns
    -------
    q1norm : 1-Norm of the Qubit Hamiltonian  
    '''
    n_orb = one_body_integrals.shape[0]
    if normal_order:
        two_body_integrals = normal_order_tbc(two_body_integrals_inp, n_orb)
    else:
        two_body_integrals = np.copy(two_body_integrals_inp)
    

    
    htilde = constant
    for p in range(n_orb):
        htilde += one_body_integrals[p,p]
        print(htilde)
        for q in range(n_orb):
                # print(two_body_integrals[p,q,q,p],two_body_integrals[p,q,p,q])
                htilde += (1/2 * two_body_integrals[p,q,q,p]) -\
                          (1/4 * two_body_integrals[p,q,p,q])
    
    htildepq = np.zeros(one_body_integrals.shape)
    for p in range(n_orb):
        for q in range(n_orb):
            htildepq[p,q] = one_body_integrals[p,q]
            for r in range(n_orb):
                htildepq[p,q] += ((two_body_integrals[p,r,r,q]) - \
                                  (1/2 * two_body_integrals[p,r,q,r]))
    
    q1norm1 = abs(htilde) + np.sum(np.absolute(np.diag(htildepq)))
    q1norm3 = 0
    for p in range(n_orb):
        for q in range(n_orb):
            if p != q:
                q1norm3 += abs(htildepq[p,q])
    q1norm2 = 0
    for p in range(n_orb):
        q1norm3 += 1/4 * abs(two_body_integrals[p,p,p,p])
        for q in range(n_orb):
            if p != q:
                q1norm3 += abs(two_body_integrals[p,p,p,q])
                q1norm1 += 1/4 * abs(two_body_integrals[p,q,p,q]-\
                                      two_body_integrals[p,q,q,p])
                q1norm3 += 1/4 * abs(two_body_integrals[p,q,q,p])
                q1norm3 += 1/2 * abs(two_body_integrals[p,q,p,q])
            for r in range(n_orb):
                if p != q and q!= r and p!=r:
                    q1norm3 += 1/2 * abs(two_body_integrals[p,r,q,r]-\
                                          two_body_integrals[p,r,r,q])
                    q1norm3 += 1/2 * abs(two_body_integrals[p,r,r,q])
                    q1norm3 += abs(two_body_integrals[p,q,r,q])
                for s in range(n_orb):
                    if p>q and r>s and p!=q and p!=r and p!=s and q!=r and\
                        q!=s and r!=s:
                        q1norm2 += abs(two_body_integrals[p,q,r,s] - \
                                        two_body_integrals[p,q,s,r])
    q1norm = q1norm1 + q1norm2 + q1norm3
    return q1norm
            