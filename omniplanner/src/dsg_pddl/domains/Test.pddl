(define (domain test-domain)
    (:requirements :derived-predicates :typing :adl)
    (:types
        place dsg_object region - object
    )

    (:predicates
        (object-in-place ?o - dsg_object ?p - place)
        (place-in-region ?p - place ?r - region)
        (connected ?s - place ?t - place)

        (at-poi ?p - place)
        (at-place ?p - place)
        (at-object ?o - dsg_object)
        (in-region ?r - region)

        (visited-place ?p - place)
        (visited-object ?o - dsg_object)
        (visited-region ?r - region)

        (hand-full)
        (holding ?o - dsg_object)

        (suspicious ?o - dsg_object)
        (safe ?o - dsg_object)
        (unsafe-place ?p - place)
    )

    (:functions
        (distance ?s ?t)
        (total-cost)
    )

    (:derived (at-place ?p - place)
        (at-poi ?p))

    (:derived (at-object ?o - dsg_object)
        (exists (?p - place) (and (at-poi ?p) (object-in-place ?o ?p))))

    (:derived (in-region ?r - region)
        (exists (?p - place) (and (at-poi ?p) (place-in-region ?p ?r))))

    (:derived (visited-object ?o - dsg_object)
        (exists (?p - place) (and (visited-place ?p) (object-in-place ?o ?p))))

    (:derived (visited-region ?r - region)
        (exists (?p - place) (and (visited-place ?p) (place-in-region ?p ?r))))

    (:derived (safe ?o - dsg_object)
        (not (suspicious ?o)))

    (:action goto-poi
        :parameters (?s - place ?t - place)
        :precondition (and (at-poi ?s)
                           (or (connected ?s ?t)
                               (connected ?t ?s))
                           (not (unsafe-place ?t)))
        :effect (and (not (at-poi ?s))
                     (at-poi ?t)
                     (visited-place ?t)
                     (increase (total-cost) (distance ?s ?t)))
    )

    (:action pick-object
     :parameters (?o - dsg_object ?p - place)
     :precondition (and (not (hand-full))
                        (safe ?o)
                        (at-poi ?p)
                        (object-in-place ?o ?p))
     :effect (and (holding ?o)
                  (hand-full)
                  (not (object-in-place ?o ?p)))
    )

    (:action place-object
     :parameters (?o - dsg_object ?p - place)
     :precondition (and (holding ?o)
                        (at-poi ?p))
     :effect (and (not (holding ?o))
                  (not (hand-full))
                  (object-in-place ?o ?p))
    )

    (:action inspect
     :parameters (?o - dsg_object ?p - place ?t - place)
     :precondition (and (suspicious ?o)
                        (object-in-place ?o ?p)
                        (at-place ?t)
                        (or (connected ?p ?t)
                            (connected ?t ?p)))
     :effect (and (not (suspicious ?o))
                  (not (unsafe-place ?p))
                  (increase (total-cost) 10))
    )

)
